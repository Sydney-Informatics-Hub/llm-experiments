""" Reprompting paper

Algo.
    1. init with a set of CoT examples.
    2. for each example:
        2a. sample using LLM_1 (zero-shot) for steps + answer.
        2b. if answers are equivalent, then we replace the sampled steps.
        note: incorrect answer is sometimes from based on a uniform distribution. (prob threshold)
    3. we now have N number of examples.
    4. Sample again (few-shot this time, K-shot)
        4a. select one of the N examples.
        4b. select a K number of examples excluding the one we just sampled.
        4c. sample using LLM_2 (K-shot) for steps + answer.
        4d. if answers (selecte v.s. sampled) are equivalent,
                then we replace the sampled steps with the selected.
    repeat until convergence OR M iterations reached.

Why do we keep incorrect answers?
"""
from llm_experiments.cot.cot import COT_EXAMPLE
from llm_experiments.cot.samplers import BaseCoTSampler


class Reprompting(BaseCoTSampler):
    def __init__(self, llm_1, llm_2):
        super(BaseCoTSampler).__init__()
        self.llm_1 = llm_1
        self.llm_2 = llm_2

    @property
    def sample_pool(self) -> list[COT_EXAMPLE]:
        return list()

    def init(self, *args, **kwargs):
        pass

    def sample(self, n: int) -> list[COT_EXAMPLE]:
        pass
