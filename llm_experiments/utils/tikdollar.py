""" Token Counter

OpenAI charges you for both input and output tokens.

1. Before you send: Count number of input tokens.
2. Before you send: Estimate number of output tokens.
3. After you send: Update actual number of output tokens.

Useful features:
1. cost threshold - before you send + cut off automatically with internal state counter.
"""
import sys
from typing import Callable, NamedTuple
import functools
from dataclasses import dataclass

from langchain.llms import OpenAI, OpenAIChat
from langchain.callbacks import get_openai_callback
from langchain.callbacks.openai_info import (
    OpenAICallbackHandler,
    get_openai_token_cost_for_model,
)

import tiktoken


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

    def decorator(func: Callable):
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
            if not isinstance(llm, OpenAI) and not isinstance(llm, OpenAIChat):
                raise TypeError("llm is not OpenAI or OpenAIChat langchain models.")
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a str. If you're using a prompt template, call .format() first.")

            # todo: estimate cost from input tokens, optionally estimate output tokens.
            num_input_tokens: int
            num_output_tokens: int
            cost: float

            num_input_tokens = count_input_tokens(llm.model_name, prompt)
            cost = get_openai_token_cost_for_model(
                model_name=llm.model_name, num_tokens=num_input_tokens,
                is_completion=False
            )

            min_cost_after_call = tikdollar_wrapper.tikdollar.cost + cost
            if verbose:
                print(f"Estimated cost on next request: ${cost:.8f}\tAccumulates to: ${min_cost_after_call:.8f}")
            if min_cost_after_call > cost_threshold:
                print(f"Estimated cost exceeds threshold: {min_cost_after_call} > {cost_threshold}.", file=sys.stderr)
                if raise_err:
                    raise CostThresholdReachedException(cost_threshold=cost_threshold, cost=min_cost_after_call)

            cb: OpenAICallbackHandler
            with get_openai_callback() as cb:
                res = func(*args, **kwargs)
                if num_input_tokens != cb.prompt_tokens:
                    print(f"[warn] estimated input tokens did not match with callback count. "
                          f"Expecting {num_input_tokens}. Got {cb.prompt_tokens}",
                          file=sys.stderr)
                num_input_tokens = cb.prompt_tokens
                num_output_tokens = cb.completion_tokens
                cost = cb.total_cost
                if verbose: print(f"Actual cost on request: "
                                  f"Input tokens={num_input_tokens}\t"
                                  f"Output tokens={num_output_tokens}\t"
                                  f"Cost={cost}")
            if min_cost_after_call > cost_threshold:
                raise CostThresholdReachedException(cost_threshold=cost_threshold,
                                                    cost=min_cost_after_call)

            cb: OpenAICallbackHandler
            with get_openai_callback() as cb:
                res = func(*args, **kwargs)
                if num_input_tokens != cb.prompt_tokens:
                    print("[warn] estimated input tokens count does not match with callback count.", file=sys.stderr)
                num_input_tokens = cb.prompt_tokens
                num_output_tokens = cb.completion_tokens
                cost = cb.total_cost
                tikdollar_wrapper.tikdollar.num_input_tokens += num_input_tokens
                tikdollar_wrapper.tikdollar.num_output_tokens += num_output_tokens
                tikdollar_wrapper.tikdollar.cost += cost
            return res

        tikdollar_wrapper.tikdollar = TikDollar.empty()
        return tikdollar_wrapper

    return decorator


@tikdollar(cost_threshold=1.0, repeat=0, verbose=False)
def test_tikdollar(llm: OpenAI, prompt: str):
    return llm.generate([prompt])


if __name__ == '__main__':
    # todo: ltest - OpenAI
    # todo: ltest - ChatOpenAI
    openai = OpenAI(model_name='text-ada-001', n=1)
    # openaichat = OpenAIChat(model_name='gpt3.5-turbo')

    prompt = 'hello, repeat after me 1 time.'
    res = test_tikdollar(openai, prompt=prompt)
    print(test_tikdollar.tikdollar)

    print("\n---\n")
    print(res)
