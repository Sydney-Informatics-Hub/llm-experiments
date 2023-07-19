""" Samplers

Weighted: Given weights to each example, sample examples.
Random: Inherits Weighted but weights are predefined to be uniform across examples.
Gibbs:

Is there a way to make the sampler sample more from the input of various weights?

"""

from abc import ABCMeta, abstractmethod
from llm_experiments.cot.cot import COT_EXAMPLE


class BaseCoTSampler(ABCMeta):

    @property
    @abstractmethod
    def sample_pool(self) -> list[COT_EXAMPLE]:
        raise NotImplementedError()

    @abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError("Base init method is not implemented.")

    @abstractmethod
    def sample(self, n: int) -> list[COT_EXAMPLE]:
        if not isinstance(n, int): raise TypeError("n must be an integer.")
        raise NotImplementedError("Base sample method is not implemented.")


def sample_til(sampler: BaseCoTSampler, n: int, max_input_tokens: int) -> list[COT_EXAMPLE]:
    """ Keep on sampling until it is under the threshold of max_input_tokens. """
    pass  # todo


def random_sample(cot_examples: list[COT_EXAMPLE], n: int) -> list[COT_EXAMPLE]:
    import numpy as np
    indices = np.random.choice(np.arange(len(cot_examples)), size=n, replace=False)
    return np.array(cot_examples)[indices].tolist()
