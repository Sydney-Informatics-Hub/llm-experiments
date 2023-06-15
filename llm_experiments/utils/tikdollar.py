""" TikDollar

OpenAI charges you for both input and output tokens.

1. Before you send: Count number of input tokens.
2. Before you send: Estimate number of output tokens.
3. After you send: Update actual number of output tokens.

Features:
1. cost threshold - before you send + cut off automatically with internal state counter.
"""
import sys
from typing import Callable, Union, Any
import functools
from dataclasses import dataclass

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.openai_info import (
    get_openai_token_cost_for_model,
)

import tiktoken
from langchain.schema import LLMResult

PROMPT = str
ERR_SUPPORTED_MODELS = "Currently only supports OpenAI and ChatOpenAI llms."


class CostThresholdReachedException(Exception):
    def __init__(self, cost: float, cost_threshold: float):
        self.cost, self.cost_threshold = cost, cost_threshold

    def __repr__(self) -> str:
        return f"<cost={self.cost} cost_threshold={self.cost_threshold}>"


@dataclass
class TikDollar(object):
    num_input_tokens: int
    num_output_tokens: int
    cost: float

    @property
    def total_tokens(self) -> int:
        return self.num_input_tokens + self.num_output_tokens

    @classmethod
    def empty(cls):
        return cls(num_input_tokens=0, num_output_tokens=0, cost=0)


# low level function
def count_input_tokens(model: str, prompt: str) -> int:
    if not isinstance(model, str): raise TypeError("model must be a string.")
    if not isinstance(prompt, str): raise TypeError("prompt must be a string.")
    encoder = tiktoken.encoding_for_model(model_name=model)
    tokens = encoder.encode(prompt)
    return len(tokens)


def tikdollar(cost_threshold: float, raise_err: bool = True, verbose: bool = False):
    if isinstance(cost_threshold, int): cost_threshold = float(cost_threshold)
    if not isinstance(cost_threshold, float): raise TypeError("cost_threshold must be a float.")
    if not cost_threshold >= 0: raise ValueError("cost_threshold must be >= 0.")
    if not isinstance(raise_err, bool): raise TypeError("raise_err must be a boolean.")
    if not isinstance(verbose, bool): raise TypeError("verbose must be a boolean.")

    cost_threshold: float = cost_threshold
    verbose = verbose
    raise_err = raise_err

    def decorator(func: Callable[[Union[OpenAI, ChatOpenAI], PROMPT, Any], LLMResult]):
        @functools.wraps(func)
        def tikdollar_wrapper(*args, **kwargs):
            if len(args) >= 2:
                llm = args[0]
                prompt = args[1]
            elif len(args) == 1:
                llm = args[0]
                prompt = kwargs.get('prompt', None)
            else:
                llm = kwargs.get('llm', None)
                prompt = kwargs.get('prompt', None)
            if llm is None or prompt is None: raise ValueError("Arguments llm or prompt is missing.")
            if not isinstance(llm, OpenAI) and not isinstance(llm, ChatOpenAI):
                raise TypeError("llm is not OpenAI or ChatOpenAI langchain models.")
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a str. If you're using a prompt template, call .format() first.")

            num_input_tokens: int
            num_output_tokens: int
            cost: float

            num_input_tokens = count_input_tokens(llm.model_name, prompt)
            cost = get_openai_token_cost_for_model(
                model_name=llm.model_name, num_tokens=num_input_tokens,
                is_completion=False
            )
            if isinstance(llm, OpenAI):
                est_output_tokens = int(llm.max_tokens / 2)
                num_output_tokens = est_output_tokens * max(llm.n, llm.best_of)  # n completions
            elif isinstance(llm, ChatOpenAI):
                est_output_tokens = 0
                num_output_tokens = est_output_tokens * llm.n  # n completions
            else:
                raise NotImplementedError(ERR_SUPPORTED_MODELS)
            cost += get_openai_token_cost_for_model(
                model_name=llm.model_name, num_tokens=num_output_tokens,
                is_completion=True,
            )

            min_cost_after_call = tikdollar_wrapper.tikdollar.cost + cost
            if verbose:
                print(f"{'Estimated cost on next request:'.ljust(40)} "
                      f"Input = {num_input_tokens} tokens\t"
                      f"Output = {num_output_tokens} tokens\t"
                      f"${cost:.8f}\tAccumulates to: ${min_cost_after_call:.8f}")
            if min_cost_after_call > cost_threshold:
                print(f"Estimated cost exceeds threshold: ${min_cost_after_call} > ${cost_threshold}. "
                      f"Difference = ${cost_threshold - min_cost_after_call}", file=sys.stderr)
                if raise_err:
                    raise CostThresholdReachedException(cost_threshold=cost_threshold, cost=min_cost_after_call)

            res: LLMResult = func(*args, **kwargs)
            token_usage = res.llm_output.get('token_usage')
            token_usage = res.llm_output.get('token_usage', None)
            if token_usage is None:
                raise RuntimeError("Expected LLMResult.llm_output to contain 'token_usage'.")

            num_input_tokens = token_usage.get('prompt_tokens')
            total_tokens = token_usage.get('total_tokens')
            num_output_tokens = total_tokens - num_input_tokens
            cost = get_openai_token_cost_for_model(
                model_name=llm.model_name, num_tokens=num_input_tokens, is_completion=False
            )
            cost += get_openai_token_cost_for_model(
                model_name=llm.model_name, num_tokens=num_output_tokens, is_completion=True
            )
            if verbose: print(f"{'Actual cost after request:'.ljust(40)} "
                              f"Input = {num_input_tokens} tokens\t"
                              f"Output = {num_output_tokens} tokens\t"
                              f"Cost={cost}")
            tikdollar_wrapper.tikdollar.num_input_tokens += num_input_tokens
            tikdollar_wrapper.tikdollar.num_output_tokens += num_output_tokens
            tikdollar_wrapper.tikdollar.cost += cost
            return res

        tikdollar_wrapper.tikdollar = TikDollar.empty()
        return tikdollar_wrapper

    return decorator


@tikdollar(cost_threshold=0.000001, verbose=True, raise_err=False)
def test_tikdollar(llm: Union[OpenAI, ChatOpenAI], prompt: str) -> LLMResult:
    if isinstance(llm, OpenAI):
        return llm.generate([prompt])
    elif isinstance(llm, ChatOpenAI):
        # todo: uses messages.
        from langchain.schema import (
            AIMessage, HumanMessage, SystemMessage
        )
        return llm.generate([[HumanMessage(content=prompt)]])
    else:
        raise NotImplementedError(ERR_SUPPORTED_MODELS)


if __name__ == '__main__':
    # openai = OpenAI(model_name='text-ada-001', n=1)
    openai = ChatOpenAI(model_name='gpt-3.5-turbo')

    prompts = ['hello, repeat after me 1 time.',
               'hello, repeat after me 2 times.']

    for prompt in prompts:
        print(prompt)
        res = test_tikdollar(openai, prompt=prompt)
        print(test_tikdollar.tikdollar)
