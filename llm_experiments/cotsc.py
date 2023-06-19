""" Chain-of-Thought Self Consistency

In simple terms, use a probabilistic sampling strategy for LLM decoding.
Take the majority vote of the answer.

Sampling Strategies:
1. temperature sampling - softmax doesn't work well with large values. Increase for more stochastic outputs.
2. top-k - select top-k most probable tokens and sample from them.
3. nucleus - instead of top-k, use a probability threshold.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from langchain.schema import Generation, OutputParserException
from langchain.llms import OpenAI
from langchain.schema import LLMResult
from llm_experiments.utils import prompt_check


@dataclass
class SamplingScheme(object):
    temperature: float
    top_k: Optional[int]
    top_p: float  # i.e. nucleus sampling

    def openai(self) -> dict:
        # there is no top_k parameter for openai api.
        if self.top_k is not None:
            print("-- OpenAI does not have top_k. It is ignored.", file=sys.stderr)

        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top p must be between 0 and 1. "
                             "It is the probability mass of output tokens to sample from.")
        return {
            'temperature': self.temperature,
            'top_p': self.top_p
        }


# Configurations used in the CoT-SC paper:
UL2_20B_LaMDA_137B = SamplingScheme(temperature=0.5, top_k=40, top_p=1)
PaLM_540B = SamplingScheme(temperature=0.7, top_k=40, top_p=1)
GPT3 = SamplingScheme(temperature=0.7, top_k=40, top_p=1)
TEST = SamplingScheme(temperature=0.5, top_k=None, top_p=0.5)

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

COT_EXAMPLE = dict[str, str]

cot_prompt_template = PromptTemplate(
    input_variables=['query', 'steps', 'answer'],
    template="Query: {query}\nAnswer: {steps}. The answer is {answer}."
)


def create_cot_prompt_example(
        query: str,
        steps: str,
        answer: str,
) -> COT_EXAMPLE:
    return {
        'query': query,
        'steps': steps,
        'answer': answer,
    }


def create_cot_prompt_template(
        instructions: Optional[str],
        cot_examples: list[COT_EXAMPLE],
) -> FewShotPromptTemplate:
    """ Create a prompt template following Chain of Thoughts. """
    if not isinstance(instructions, str) and instructions is not None:
        raise TypeError("instructions must be a string.")
    if isinstance(instructions, str) and not instructions.strip().endswith("\n"):
        instructions += "\n"
    instructions = instructions.strip()

    if not isinstance(cot_examples, list): raise TypeError("cot_examples must be a list.")
    if len(cot_examples) <= 0: raise ValueError("There must be at least 1 cot_example.")

    for ex in cot_examples:
        if not {"query", "steps", "answer"} == set(ex.keys()):
            raise ValueError(f"Missing keys (query, steps, answer) in example: {ex}")

    return FewShotPromptTemplate(
        prefix=instructions,
        example_prompt=cot_prompt_template,
        examples=cot_examples,
        suffix="Query: {query}",
        input_variables=['query'],
        example_separator='\n',
    )


NUM_VOTES = int
STEPS = set[str]
CLAZZ = str
VOTES = list[tuple[CLAZZ, dict[str, Union[NUM_VOTES, STEPS]]]]


class ClassificationOutput(BaseModel):
    answer: str = Field(description="the classification")
    steps: str = Field(description="the reasoning steps that resulted in the classification")


class CoTSC(object):
    MODELS = ('text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001')

    def __init__(self,
                 model: str,
                 prompt: FewShotPromptTemplate,
                 classes: list[str],
                 sampling_scheme: SamplingScheme,
                 n_completions: int,
                 ):
        if not isinstance(model, str): raise TypeError(f"model must be a string. {', '.join(CoTSC.MODELS)}")
        if model not in CoTSC.MODELS: raise ValueError(f"model must be one of {', '.join(CoTSC.MODELS)}")
        if not isinstance(prompt, FewShotPromptTemplate): raise TypeError("prompt must be a FewShotPrompt for CoT.")
        if not prompt.suffix: raise ValueError("prompt must have a suffix for CoT.")
        if not isinstance(classes, list): raise TypeError("classes must be a list of strings.")
        if len(classes) <= 0: raise ValueError("There must be at least one class.")
        if not isinstance(classes[0], str): raise TypeError("classes must be a list of strings.")
        if not isinstance(n_completions, int): raise TypeError("n_completions must be an integer.")
        if not n_completions > 0: raise ValueError("n_completions must be > 0.")
        if n_completions > 10: print(f"Warning: {n_completions=}. This may incur significant costs.", file=sys.stderr)

        self.llm = OpenAI(
            model_name=model,
            n=n_completions,
            **sampling_scheme.openai(),
        )
        self.model = model
        self.classes = classes
        self.n_completions = n_completions

        # output defs
        parser = PydanticOutputParser(pydantic_object=ClassificationOutput)  # note: hard coded output definition
        prompt.suffix = "\n\n{format_instructions}\n\n" + prompt.suffix
        prompt.partial_variables = {'format_instructions': parser.get_format_instructions()}
        self.prompt = prompt
        self.parser = parser

    def run(self, query: str) -> VOTES:
        # use the few shot prompt as input to the llm.
        # generate X number of outputs.
        # parse the output for answer.
        prompt = self.prompt.format(query=query)
        prompts = list(map(prompt_check, [prompt]))
        output = self._tikdollar_run(self.llm, prompts[0])
        if not len(output.generations) == 1: raise RuntimeError("1 prompt is provided, expecting 1 output generation")
        completions = output.generations[0]
        if not len(completions) == self.n_completions:
            raise RuntimeError(f"Expecting {self.n_completions}. Got {len(completions)}")

        return self._majority_vote([self.fallback_parse(c) for c in completions])

    @staticmethod
    def _tikdollar_run(llm, prompt) -> LLMResult:
        return llm.generate([prompt])

    @staticmethod
    def _majority_vote(parsed: list[ClassificationOutput]) -> VOTES:
        """ Collate votes on classification based on LLM outputs. """
        votes = dict()
        for p in parsed:
            votes_ans = votes.get(p.answer, {'votes': 0, 'steps': set()})
            votes_ans['votes'] = votes_ans.get('votes') + 1
            votes_ans['steps'].add(p.steps)
            votes[p.answer] = votes_ans
        return sorted(votes.items(), key=lambda kv: kv[1].get('votes'), reverse=True)

    def fallback_parse(self, completion: Generation) -> ClassificationOutput:
        """ Parses the output using the parser first. Fallback to regex. Then N/A. """
        try:
            parsed = self.parser.parse(completion.text)
            return parsed
        except OutputParserException as ope:
            ans_ptn = re.compile("(" + "|".join(self.classes) + ")", flags=re.IGNORECASE)
            answers = ans_ptn.findall(completion.text)
            ans = ', '.join(answers)
            steps: str = ans_ptn.sub('', completion.text)
            steps = re.sub('answer[:]?', '', steps, flags=re.IGNORECASE)
            steps = steps.strip()
            # todo: log
            return ClassificationOutput(answer=ans, steps=steps)
        except Exception as e:
            # todo: log
            return ClassificationOutput(answer='N/A', steps='N/A')

    @classmethod
    def from_toml(cls,
                  model: str,
                  prompt_toml: Union[str, Path],
                  sampling_scheme: SamplingScheme,
                  n_completions: int):
        """ Create the appropriate prompt template based on TOML file setting. """
        prompt_toml = Path(prompt_toml)
        if not prompt_toml.suffix == '.toml': raise ValueError("prompt_toml is not a toml file.")
        import toml
        data = toml.load(prompt_toml)
        classes = list(data.keys())

        instructions = []
        cot_examples = []
        for clz in classes:
            for example in data.get(clz).get('examples', list()):
                cot_ex = create_cot_prompt_example(
                    query=example.get('query'),
                    steps=example.get('steps'),
                    answer=clz
                )
                cot_examples.append(cot_ex)

            instruction = data.get(clz).get('instruction')
            instruction = f"{clz}: {instruction}"
            instructions.append(instruction)

        instruction = f"""The following are {len(classes)} classes with a description of each. 
    Please classify each 'query' as one of the {len(classes)} classes.\n""" + '\n'.join(instructions) + "\n\n"

        template = create_cot_prompt_template(
            instructions=instruction,
            cot_examples=cot_examples,
        )
        return cls(model=model,
                   prompt=template,
                   classes=classes,
                   sampling_scheme=sampling_scheme,
                   n_completions=n_completions)


if __name__ == '__main__':
    from pprint import pprint
    import argparse

    parser = argparse.ArgumentParser(description='chain of thoughts (self consistency) - classification')
    parser.add_argument('--prompt-toml', type=str, help="the path to the toml prompt path.")
    args = parser.parse_args()

    assert Path(args.prompt_toml).exists(), f"{args.prompt_toml} does not exist."

    cotsc = CoTSC.from_toml(model='text-davinci-003',
                            prompt_toml=args.prompt_toml,
                            sampling_scheme=SamplingScheme(temperature=1.0, top_p=1, top_k=None),
                            n_completions=5)

    while (q := input("Enter a query (.quit to quit): ")) != ".quit":
        print("query: " + q)
        votes = cotsc.run(query=q)
        pprint(votes)
