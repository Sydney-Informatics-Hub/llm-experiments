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

TEST = SamplingScheme(temperature=0.5, top_k=None, top_p=1)

if __name__ == '__main__':
    # 1. model.generate n completions.
    # 2. take majority vote.

    model = OpenAI(
        # model_name='text-davinci-003',
        # model_name='text-davinci-002',
        model_name='text-curie-001',
        # model_name='text-babbage-001',
        # model_name='text-ada-001',
        n=(num_completions := 3),
        max_tokens=256,  # default. may be set to 1 for classification.
        **TEST.openai(),
    )
    print(model)

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
