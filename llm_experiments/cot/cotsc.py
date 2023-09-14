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
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, BaseMessagePromptTemplate
from langchain.schema import LLMResult, BaseMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from llm_experiments.utils import prompt_check
from llm_experiments.cot.cot import (create_cot_prompt_template,
                                     create_cot_prompt_example,
                                     COT_TEMPLATE, CoTDataLeak, CoT)

__all__ = ['SamplingScheme', 'CoTSC']


def flatten(nested: list) -> list:
    flattened = []
    for e in nested:
        if isinstance(e, list):
            flattened.extend(flatten(e))
        else:
            flattened.append(e)
    return flattened


@dataclass
class SamplingScheme(object):
    temperature: float
    top_p: float  # i.e. nucleus sampling
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    def openai(self) -> dict:
        # there is no top_k parameter for openai api.
        if self.top_k is not None:
            print("-- OpenAI does not have top_k. It is ignored.", file=sys.stderr)

        # compulsory
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top p must be between 0 and 1. "
                             "It is the probability mass of output tokens to sample from.")
        args = {
            'temperature': self.temperature,
            'top_p': self.top_p
        }
        # optional
        if self.presence_penalty is not None:
            if not -2.0 <= self.presence_penalty <= 2.0:
                raise ValueError("Presence penalty must be between -2.0 and 2.0. "
                                 "It is how much to penalise the token if it's already been sampled.")
            args['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty is not None:
            if not -2.0 <= self.presence_penalty <= 2.0:
                raise ValueError("Frequency penalty must be between -2.0 and 2.0. "
                                 "It is how much to penalise the token if it's already been sampled proportional to "
                                 "the number of times its been sampled.")
            args['frequency_penalty'] = self.frequency_penalty
        return args


# Configurations used in the CoT-SC paper:
UL2_20B_LaMDA_137B = SamplingScheme(temperature=0.5, top_k=40, top_p=1)
PaLM_540B = SamplingScheme(temperature=0.7, top_k=40, top_p=1)
GPT3 = SamplingScheme(temperature=0.7, top_k=40, top_p=1)
TEST = SamplingScheme(temperature=0.5, top_k=None, top_p=0.5)

NUM_VOTES = int
STEPS = set[str]
CLAZZ = str
VOTES = dict[CLAZZ, dict[str, Union[NUM_VOTES, STEPS]]]
PROMPT = Union[str, BaseMessage]


class ClassificationOutput(BaseModel):
    answer: str = Field(description="the classification")
    steps: str = Field(description="the reasoning steps that resulted in the classification")


class CoTSC(object):
    MODELS = ('text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001',
              'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4')

    def __init__(self,
                 model: str,
                 prompt: COT_TEMPLATE,
                 classes: list[str],
                 sampling_scheme: SamplingScheme,
                 n_completions: int,
                 ):
        if not isinstance(model, str): raise TypeError(f"model must be a string. {', '.join(CoTSC.MODELS)}")
        if model not in CoTSC.MODELS: raise ValueError(f"model must be one of {', '.join(CoTSC.MODELS)}")
        if not isinstance(prompt, COT_TEMPLATE): raise TypeError("prompt must be a FewShotPrompt/COT_TEMPLATE for CoT.")
        if not prompt.suffix: raise ValueError("prompt must have a suffix for CoT.")
        if not isinstance(classes, list): raise TypeError("classes must be a list of strings.")
        if len(classes) <= 0: raise ValueError("There must be at least one class.")
        if not isinstance(classes[0], str): raise TypeError("classes must be a list of strings.")
        if not isinstance(n_completions, int): raise TypeError("n_completions must be an integer.")
        if not n_completions > 0: raise ValueError("n_completions must be > 0.")
        if n_completions > 10: print(f"Warning: {n_completions=}. This may incur significant costs.", file=sys.stderr)

        # output defs
        parser = PydanticOutputParser(pydantic_object=ClassificationOutput)  # note: hard coded output definition
        prompt.input_variables.append("format_instructions")
        prompt.suffix = "\n\n{format_instructions}\n\n" + prompt.suffix
        self.prompt = prompt
        self.parser = parser

        if model in ('gpt-3.5-turbo', 'gpt-4'):
            model_kwargs = sampling_scheme.openai()
            del model_kwargs['temperature']
            self.llm = ChatOpenAI(
                model_name=model,
                n=n_completions,
                temperature=sampling_scheme.temperature,
                model_kwargs=model_kwargs,
            )
            # todo: perhaps there is a better way instead of re-introducing the input variables?
            input_kwargs = {inp: f"{{{inp}}}" for inp in self.prompt.input_variables}
            self.prompt = HumanMessagePromptTemplate.from_template(
                self.prompt.format(**input_kwargs, )
            )
            self.prompt.prompt.partial_variables = {"format_instructions": self.parser.get_format_instructions()}
        else:
            self.llm = OpenAI(
                model_name=model,
                n=n_completions,
                **sampling_scheme.openai(),
            )
            self.prompt.partial_variables = {"format_instructions": self.parser.get_format_instructions()}
        self.model = model
        self.classes = classes
        self.n_completions = n_completions

        self._dataleak = CoTDataLeak(prompt, raise_err=True)

    def run(self, query: str) -> VOTES:
        # use the few shot prompt as input to the llm.
        # generate X number of outputs.
        # parse the output for answer.
        try:
            query = str(query)
        except Exception as e:
            raise TypeError(f"query must be a string. {e}")

        _ = self._dataleak.check(query)

        prompt = self.prompt.format(query=str(query))
        prompts = list(map(prompt_check, [prompt]))
        output: LLMResult = self._tikdollar_run(self.llm, prompts[0])
        if not len(output.generations) == 1: raise RuntimeError("1 prompt is provided, expecting 1 output generation")

        completions = flatten(output.generations)
        if not len(completions) == self.n_completions:
            raise RuntimeError(f"Expecting {self.n_completions}. Got {len(completions)}")

        return self._majority_vote([self.fallback_parse(c) for c in completions])

    def dryrun(self, query: str) -> PROMPT:
        try:
            query = str(query)
        except Exception as e:
            raise TypeError(f"query must be a string. {e}")

        content = self.prompt.format(query=str(query))
        if isinstance(self.prompt, BaseMessagePromptTemplate):
            return content.content
        else:
            return content

    @staticmethod
    def _tikdollar_run(llm, prompt) -> LLMResult:
        if isinstance(llm, BaseLLM):
            return llm.generate([prompt])
        elif isinstance(llm, BaseChatModel):
            messages = [[prompt]]
            return llm.generate(messages)
        else:
            raise TypeError(f"LLM must be either a {BaseLLM.__name__} or {BaseChatModel.__name__}.")

    @staticmethod
    def _majority_vote(parsed: list[tuple[ClassificationOutput, str]]) -> VOTES:
        """ Collate votes on classification based on LLM outputs. """
        votes = dict()
        for p, completion in parsed:
            votes_ans = votes.get(p.answer, {'votes': 0, 'steps': set(), 'completions': list()})
            votes_ans['votes'] = votes_ans.get('votes') + 1
            votes_ans['steps'].add(p.steps)
            votes_ans['completions'].append(completion)
            votes[p.answer] = votes_ans
        return votes

    def fallback_parse(self, completion: Generation) -> tuple[ClassificationOutput, str]:
        """ Parses the output using the parser first. Fallback to regex. Then N/A. """
        try:
            parsed: ClassificationOutput = self.parser.parse(completion.text)
            return parsed, completion.text
        except OutputParserException as ope:
            ans_ptn = re.compile("(" + "|".join(self.classes) + ")", flags=re.IGNORECASE)
            answers = ans_ptn.findall(completion.text)
            ans = ', '.join(answers)
            steps: str = ans_ptn.sub('', completion.text)
            steps = re.sub('answer[:]?', '', steps, flags=re.IGNORECASE)
            steps = steps.strip()
            return ClassificationOutput(answer=ans, steps=steps), completion.text
        except Exception as e:
            return ClassificationOutput(answer='N/A', steps=completion.text), completion.text

    @classmethod
    def from_toml(cls,
                  model: str,
                  prompt_toml: Union[str, Path],
                  sampling_scheme: SamplingScheme,
                  n_completions: int,
                  shuffle_examples: bool = True,
                  shuffle_seed: int = 42):
        """ Create the appropriate prompt template based on TOML file setting. """
        cot = CoT.from_toml(prompt_toml)
        if shuffle_examples:
            cot.shuffle_examples(seed=shuffle_seed)
        return cls.from_cot(model=model,
                            cot=cot,
                            sampling_scheme=sampling_scheme,
                            n_completions=n_completions)

    @classmethod
    def from_cot(cls,
                 model: str,
                 cot: CoT,
                 sampling_scheme: SamplingScheme,
                 n_completions: int):
        return cls(model=model,
                   prompt=cot.prompt,
                   classes=cot.classes,
                   sampling_scheme=sampling_scheme,
                   n_completions=n_completions)


from unittest import TestCase


class TestCoTSC(TestCase):
    def test_cotsc_from_cot(self):
        cot = CoT.from_toml('./notebooks/cotsc/classification.toml')
        cot.sample(method='random', n=1)
        cotsc = CoTSC.from_cot(model='gpt-3.5-turbo', cot=cot, sampling_scheme=TEST, n_completions=1)
        print(cotsc.dryrun(query='<query>'))


if __name__ == '__main__':
    from pprint import pprint
    import argparse

    parser = argparse.ArgumentParser(description='chain of thoughts (self consistency) - classification')
    parser.add_argument('--prompt-toml', required=True, type=str, help="the path to the toml prompt path.")
    parser.add_argument('--n-completions', required=True, type=int, help='model number of output completions.')
    parser.add_argument('--temperature', required=True, type=float, help='model temperature scaling')
    parser.add_argument('--top-p', required=True, type=float, help='model nucleus sampling')
    parser.add_argument('--presence-penalty', required=False, type=float, help='model presence penalty', default=None)
    args = parser.parse_args()

    assert args.n_completions <= 10, "Too many completions? Hard coded to stop at 10 to avoid excessive cost."

    assert Path(args.prompt_toml).exists(), f"{args.prompt_toml} does not exist."

    cotsc = CoTSC.from_toml(model='gpt-3.5-turbo',
                            prompt_toml=args.prompt_toml,
                            sampling_scheme=SamplingScheme(temperature=args.temperature,
                                                           top_p=args.top_p,
                                                           presence_penalty=args.presence_penalty),
                            n_completions=5)

    while (q := input("Enter a query (.quit to quit): ")) != ".quit":
        print("query: " + q)
        votes = cotsc.run(query=q)
        for clazz, data in votes.items():
            print(f"== Classification: {clazz} ==")
            steps = data.get('steps')
            for i, step in enumerate(steps):
                print(f"[Step[{i}]]: {step}")
