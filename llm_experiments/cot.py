""" Chain Of Thought
A prompting technique used to encourage the model to generate a series of intermediate reasoning steps.

This script uses examples from the Chain of Thoughts paper.
https://arxiv.org/abs/2201.11903
"""
import sys
import random

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from typing import Optional

__all__ = ['create_cot_prompt_example', 'create_cot_prompt_template',
           'COT_EXAMPLE', 'COT_TEMPLATE',
           'CoTDataLeak', 'CoTDataLeakException']

# type aliases
COT_EXAMPLE = dict[str, str]
COT_TEMPLATE = FewShotPromptTemplate

# constants
_QUERY = 'query'
_STEPS = 'steps'
_ANS = 'answer'


def create_cot_prompt_example(
        query: str,
        steps: str,
        answer: str,
) -> COT_EXAMPLE:
    return {
        _QUERY: query,
        _STEPS: steps,
        _ANS: answer,
    }


_cot_example_template = PromptTemplate(
    input_variables=[_QUERY, _STEPS, _ANS],
    template=f"Query: {{{_QUERY}}}\nAnswer: {{{_STEPS}}}. The answer is {{{_ANS}}}."
)


def create_cot_prompt_template(
        instructions: Optional[str],
        cot_examples: list[COT_EXAMPLE],
        shuffle: bool,
        seed: int = 42,
) -> COT_TEMPLATE:
    """ Create a prompt template following Chain of Thoughts. """
    if not isinstance(instructions, str) and instructions is not None:
        raise TypeError("instructions must be a string.")
    if instructions is None: instructions = ''
    if isinstance(instructions, str) and not instructions.strip().endswith("\n"):
        instructions += "\n"
    instructions = instructions.strip()

    if not isinstance(cot_examples, list): raise TypeError("cot_examples must be a list.")
    if len(cot_examples) <= 0: raise ValueError("There must be at least 1 cot_example.")

    if shuffle:
        random.seed(seed)
        random.shuffle(cot_examples)

    for ex in cot_examples:
        if not {_QUERY, _STEPS, _ANS} == set(ex.keys()):
            raise ValueError(f"Missing keys ({_QUERY}|{_STEPS}|{_ANS}) in example: {ex}")

    return FewShotPromptTemplate(
        prefix=instructions + "\n\n",
        example_prompt=_cot_example_template,
        examples=cot_examples,
        suffix=f"Query: {{{_QUERY}}}",
        input_variables=[_QUERY],
        example_separator='\n',
    )


class CoTDataLeakException(Exception):
    def __init__(self, query: str):
        self.query = query


class CoTDataLeak(object):
    """ Checks for Data Leakage given CoT examples used. """

    CHECK_TYPES = ('exact',)

    def __init__(self, template: COT_TEMPLATE, check_type: str = 'exact', raise_err: bool = True):
        if not isinstance(template, COT_TEMPLATE):
            raise TypeError("template must be a FewShotPromptTemplate/COT_TEMPLATE.")
        if not check_type in self.CHECK_TYPES:
            raise ValueError(f"check_type must be one of {', '.join(self.CHECK_TYPES)}")
        if not isinstance(raise_err, bool):
            raise TypeError("raise_err must be a boolean.")

        queries = [ex.get("query", None) for ex in template.examples]
        assert None not in queries, "Missing 'query' parameter in your CoT examples."
        assert len(queries) == len(template.examples), "Mismatched number of CoT examples of number of queries."

        self.queries = queries
        self.template = template
        self.check_type = check_type
        self.raise_err = raise_err

        self._setup: dict[str, Optional] = dict()

    def check(self, query: str) -> bool:
        """ Checks whether your query is leaked in your CoT examples."""
        if self.check_type == 'exact':
            matched = self.exact_matched(query)
        else:
            raise NotImplementedError(f"Missing implementation for {self.check_type}.")

        if matched:
            if self.raise_err: raise CoTDataLeakException(query)
            else: print(f'[warn] query is leaked as part of CoT examples. {query}', file=sys.stderr)
        return not matched

    def exact_matched(self, query: str) -> bool:
        if self._setup.get('exact', None) is None:
            self._setup['exact'] = set(self.queries)  # performance
        return query in self._setup['exact']


### Unit Tests ###
from unittest import TestCase


class TestCoT(TestCase):
    def test_create_cot_example(self):
        ex = create_cot_prompt_example('query placeholder', 'steps placeholder', 'answer placeholder')
        target = {'query': 'query placeholder', 'steps': 'steps placeholder', 'answer': 'answer placeholder'}
        assert ex == target, "COT example created is invalid."

    def test_dataleak(self):
        ex = create_cot_prompt_example('query placeholder', 'steps placeholder', 'answer placeholder')
        template = create_cot_prompt_template("", cot_examples=[ex])
        dataleak = CoTDataLeak(template=template, raise_err=False)
        assert not dataleak.check('query placeholder')


### Example used in the Chain of Thoughts Paper ###

raw_template: str = r"""
Q: {query} 
A: {answer}"""
prompt_template = PromptTemplate(
    input_variables=['query', 'answer'],
    template=raw_template,
)

suffix: str = r"""
Q: {query}
A:"""

step_by_step_suffix: str = r"""
Q: {query}
Let's think step by step.
A:"""

cot_examples = [
    {'query': 'Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. '
              'How many tennis balls does he have now?',
     'answer': 'Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. '
               'The answer is 11.'}
]

std_examples = [
    {'query': 'Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. '
              'How many tennis balls does he have now?',
     'answer': 'The answer is 11.'}
]

std_template = FewShotPromptTemplate(
    example_prompt=prompt_template,
    examples=std_examples,
    suffix=suffix,
    input_variables=['query'],
    example_separator='\n',
)

sbs_template = FewShotPromptTemplate(
    example_prompt=prompt_template,
    examples=std_examples,
    suffix=step_by_step_suffix,
    input_variables=['query'],
)

cot_template = FewShotPromptTemplate(
    example_prompt=prompt_template,
    examples=cot_examples,
    suffix=suffix,
    input_variables=['query'],
    example_separator="\n",
)

if __name__ == '__main__':
    # uses /v1/completions endpoint
    # https://platform.openai.com/docs/models/model-endpoint-compatibility

    print("This runs the examples in the CoT paper...")
    model = OpenAI(model_name='text-davinci-003',
                   # model_name='text-davinci-002',
                   # model_name='text-curie-001',
                   # model_name='text-babbage-001',
                   # model_name='text-ada-001',
                   n=1)
    print(model)

    query = 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples ' \
            'do they have?'

    print("## Chain of Thought ##")
    print(cot_template.format(query=query))
    print(model(cot_template.format(query=query)))

    print("## Step-by-Step ##")
    print(sbs_template.format(query=query))
    print(model(prompt=sbs_template.format(query=query)))

    print("## Standard Prompting ##")
    print(std_template.format(query=query))
    print(model(prompt=std_template.format(query=query)))
