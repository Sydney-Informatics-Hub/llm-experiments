""" Chain-of-Thought Self Consistency

In simple terms, use a probabilistic sampling strategy for LLM decoding.
Take the majority vote of the answer.

```
from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
        print(cb) >>> token used, successful requests, total cost.

    return result
```

Sampling Strategies:
1. temperature sampling - softmax doesn't work well with large values. Increase for more stochastic outputs.
2. top-k - select top-k most probable tokens and sample from them.
3. nucleus - instead of top-k, use a probability threshold.
"""
import re
import sys
from dataclasses import dataclass
from typing import Optional
from collections import Counter

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from prompt_check import prompt_check


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

from langchain.llms import BaseLLM
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

COT_EXAMPLE = dict[str, str]

cot_prompt_template = PromptTemplate(
    input_variables=['query', 'steps', 'answer'],
    template="Query: {query}\n{steps}. The answer is {answer}."
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


class ClassificationOutput(BaseModel):
    answer: str = Field(description="the classification")
    reason: str = Field(description="the reason for the classification")


def create_cot_prompt_template(
        instructions: Optional[str],
        cot_examples: list[COT_EXAMPLE],
        output_definition: Optional[BaseModel],
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
        if not {"query", "reason", "answer"} == set(ex.keys()):
            raise ValueError(f"Missing keys (query, reason, answer) in example: {ex}")

    return FewShotPromptTemplate(
        prefix=instructions,
        example_prompt=cot_prompt_template,
        examples=cot_examples,
        suffix="Query: {query}",
        input_variables=['query'],
        example_separator='\n',
    )


VOTES = list


class CoTSC(object):
    MODELS = ('text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001')

    def __init__(self,
                 model: str,
                 prompt: FewShotPromptTemplate,
                 sampling_scheme: SamplingScheme,
                 ):
        if not isinstance(model, str): raise TypeError(f"model must be a string. {', '.join(CoTSC.MODELS)}")
        if model not in CoTSC.MODELS: raise ValueError(f"model must be one of {', '.join(CoTSC.MODELS)}")
        if not isinstance(prompt, FewShotPromptTemplate): raise TypeError("prompt must be a FewShotPrompt for CoT.")
        if not prompt.suffix: raise ValueError("prompt must have a suffix for CoT.")

        self.model = model
        self.prompt = prompt

        self.llm = OpenAI(
            model_name=model,
            n=(n_completions := 3),
            **sampling_scheme.openai(),
        )
        self.n_completions = n_completions

        # output defs
        self.parser = PydanticOutputParser(pydantic_object=ClassificationOutput())  # note: hard coded output definition
        prompt.suffix = "{format_instructions}\n" + prompt.suffix

    def run(self, query: str) -> VOTES:
        # use the few shot prompt as input to the llm.
        # generate X number of outputs.
        # parse the output for answer.
        prompt = self.prompt.format(query=query)
        prompts = list(map(prompt_check, [prompt]))
        output = self.llm.generate(prompts)
        if not len(output.generations) == 1: raise RuntimeError("1 prompt is provided, expecting 1 output generation")
        completions = output.generations[0]
        if not len(completions) == self.n_completions:
            raise RuntimeError(f"Expecting {self.n_completions}. Got {len(completions)}")

        return self._majority_vote([self.parser.parse(c.text) for c in completions])

    @staticmethod
    def _majority_vote(parsed: list[ClassificationOutput]) -> VOTES:
        votes = dict()
        for p in parsed:
            votes_ans = votes.get(p.answer, {'votes': 0, 'reasons': set()})
            votes_ans['votes'] = votes_ans.get('votes') + 1
            votes_ans['reasons'].add(p.reason)
            votes[p.answer] = votes_ans
        return sorted(votes, key=lambda kv: kv[1].get('votes'), reverse=True)

    print("## Chain of Thought - Self Consistency.")
    from cot import cot_template, std_template, query

    prompts = [cot_template.format(query=query)]
    prompts = list(map(prompt_check, prompts))
    results = model.generate(prompts)

    answer_pattern = re.compile(r'[0-9]+')
    prompt_answers = []
    for prompt, generations in zip(prompts, results.generations):
        answers = []
        for generation in generations:
            print("-" * 100)
            print(f"Prompt: \n{prompt}")
            print(f"Output: \n{generation.text}")
            answer = answer_pattern.findall(generation.text)[-1]
            answers.append(answer)
        prompt_answers.append(answers)
        print("=" * 200)

    print("Self Consistency: Majority vote")
    for prompt, answers in zip(prompts, prompt_answers):
        print(prompt)
        counts = Counter(answers)
        answer = counts.most_common(1)
        rest = counts.most_common(5)[1:]
        print(f"Answer = {answer}")
        print(f"Other candidates: {rest}")
