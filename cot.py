""" Chain Of Thought
A prompting technique used to encourage the model to generate a series of intermediate reasoning steps.
"""
from pprint import pprint
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI

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

query = 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples ' \
        'do they have?'
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

    print("## Chain of Thought ##")
    print(cot_template.format(query=query))
    print(model(cot_template.format(query=query)))

    print("## Step-by-Step ##")
    print(sbs_template.format(query=query))
    print(model(prompt=sbs_template.format(query=query)))

    print("## Standard Prompting ##")
    print(std_template.format(query=query))
    print(model(prompt=std_template.format(query=query)))
