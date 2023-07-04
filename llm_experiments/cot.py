""" Chain Of Thought
A prompting technique used to encourage the model to generate a series of intermediate reasoning steps.

This script uses examples from the Chain of Thoughts paper.
https://arxiv.org/abs/2201.11903
"""
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from typing import Optional

__all__ = ['create_cot_prompt_example', 'create_cot_prompt_template',
           'COT_EXAMPLE', 'COT_TEMPLATE']

COT_EXAMPLE = dict[str, str]
COT_TEMPLATE = FewShotPromptTemplate

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

    for ex in cot_examples:
        if not {"query", "steps", "answer"} == set(ex.keys()):
            raise ValueError(f"Missing keys (query, steps, answer) in example: {ex}")

    return FewShotPromptTemplate(
        prefix=instructions + "\n\n",
        example_prompt=cot_prompt_template,
        examples=cot_examples,
        suffix="Query: {query}",
        input_variables=['query'],
        example_separator='\n',
    )


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
