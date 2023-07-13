""" Tree of Thoughts

For complex tasks that require exploration or strategic lookahead, traditional or simple prompting
strategies fall short.

There are actually 2 variants ToT from:
1. Tree of Thoughts: Deliberate Problem-Solving with Large Language Models https://arxiv.org/abs/2305.10601
2. Large Language Model Guided Tree-of-Thought https://arxiv.org/abs/2305.08291

The difference is that 2. uses a 'ToT Controller' that leverages reinforcement learning. This means
that the ToT can continue to evolve and learn new knowledge even with a fixed LLM.

Benefits of ToT:
1. Modularity - thought decomposition, generation, evaluation and search can all be varied independently.
2. Generality - Standard, CoT, CoT-SC, self-refinement are special cases of ToT (i.e. trees of limited breadth/depth)
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional, Union
from collections import namedtuple
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain.prompts import (PromptTemplate,
                               FewShotPromptTemplate,
                               HumanMessagePromptTemplate)
from langchain.schema import LLMResult

from llm_experiments.cot import (CoTTemplate,
                                 create_cot_prompt_template,
                                 create_cot_prompt_example)
from llm_experiments.cotsc import SamplingScheme, flatten
import logging

logger = logging.getLogger(__name__)

Paper_Sampling_Scheme = SamplingScheme(temperature=0.7, top_p=1.0)


@dataclass
class ToTConfig:
    thoughts: int  # or steps
    candidates: int
    best: int  # how many candidates to keep.


class Thought(object):
    pass


class Heuristic(object):
    pass


""" note: Sequence of Operation
'aim to promote correct partial solutions that can be verdicted within few lookahead trials, 
and eliminate impossible partial solutions based on "too big/small" common sense".

1. Decompose -> into thoughts/steps.                                (problem specific)
2. Generate -> given states, give potential thoughts/steps.         (sample or propose)
        sample works better when the thought space is rich. e.g. for creative writing.
        propose works better when the thought space is constrained. e.g. for Game of 24, Crosswords.
3. Evaluate -> V(LLM, S). given states, evaluate them.              (value or vote)
        value ->    ask LLM to rate each isolated thought either definite/likely/impossible.
        vote ->     ask LLM to vote across thoughts for the best one. (similar in spirit to self consistency)
A search strategy. (BFS, DFS, future: A*, MCTS)

What I need:
state - input, sequence of thoughts so far.     (this is a partial solution)
thought - akin to completions in practical terms.

GameOf24:
A mathematical reasoning challenge, use 4 numbers and basic arithmetics to obtain 24. e.g. Given "4 9 10 13" solution is (10-4)*(13-9)=24.

+ How to decompose: select 2 numbers, do the arithmetic and then identify the numbers that are 'remaining'. 
+ How to generate: given (state = given 10

1. start with a propose prompt
    e.g. input: sentence; [user-defined] possible next steps: it does not blah blah.        -> generate 'Thoughts'
2. design a value prompt
    e.g. [user-defined] evaluate if blah blah -> [user-defined] sure/likely/impossible.     -> evaluate each 'Thought'
        [user-defined] (this is sample 3 times for each thought, similar to self-consistency.)
3. now we end up with X number of thoughts and their associated Value/Vote.
    select the best-B candidates.
"""

ProposeTemplate = FewShotPromptTemplate

Value = int
Vote = int
V = Union[Value, Vote]


class Values(dict):
    """ Represents the value evaluated by the LLM as string
    and their respective integer score.
    """

    def __setitem__(self, key: str, value: Value):
        # todo: restrict this to str for keys and integer for values ONLY.
        super().__setitem__(key, value)


# if there are multiple possibilities for your input.
THOUGHT_DECOMPOSITION_PROMPT: str = "Possible next steps:"


def create_propose_prompt(instruction: str, inp: str, next_steps: Optional[list[str]]
                          ) -> FewShotPromptTemplate:
    instructions = instruction + "\n" \
                   + inp + "\n" + \
                   THOUGHT_DECOMPOSITION_PROMPT + "\n"

    decompose_prompt = PromptTemplate(
        input_variables=['next_step'],
        template="{next_step}"
    )
    return FewShotPromptTemplate(
        prefix=instructions,
        example_prompt=decompose_prompt,
        examples=[{'next_step': ns} for ns in next_steps],
        suffix="Input: {input}\n" + THOUGHT_DECOMPOSITION_PROMPT + "\n",
        input_variables=['input'],
        example_separator='\n',
    )


class ThoughtGenerator(object):
    def __init__(self, llm: ChatOpenAI,  # todo: add support for OpenAI as well (simplified for the time being)
                 prompt_template: Union[CoTTemplate, ProposeTemplate],
                 n_thoughts: int,
                 ):
        self.llm: ChatOpenAI = llm
        self.n_thoughts = n_thoughts

        # validate propose template: number of thoughts should have at least 2 thoughts
        if isinstance(prompt_template, ProposeTemplate):
            if len(prompt_template.examples) < 2:
                raise ValueError("When using propose template, please provide at minimum 2 examples.")

        self.template = prompt_template

    def generate(self, **kwargs) -> list[Thought]:
        if isinstance(self.template, CoTTemplate):
            return self._generate_with_cot(**kwargs)
        elif isinstance(self.template, ProposeTemplate):
            return self._generate_with_propose(**kwargs)
        else:
            raise NotImplementedError("Only CoT and Propose templates are supported.")

    def _generate_with_cot(self, **kwargs) -> list[Thought]:
        # CoT template should use n_completions for the thought generation.
        n_completions = self.n_thoughts
        self.llm.n = n_completions

        input_kwargs = {inp: f"{{{inp}}}" for inp in self.template.input_variables}
        template = HumanMessagePromptTemplate.from_template(
            self.template.format(**input_kwargs, )
        )
        message = template.format(**kwargs)
        output: LLMResult = self.llm.generate([[message]])
        if not len(output.generations) == 1: raise RuntimeError("1 prompt provided, expecting 1 output generated.")
        completions = flatten(output.generations)
        if not len(completions) == self.llm.n:
            raise RuntimeError(f"Expecting {self.llm.n} completions. Got {len(completions)}.")
        return [c.text for c in completions]

    def _generate_with_propose(self, **kwargs) -> list[Thought]:
        # Propose template should have 1 completion and just break it down.
        n_completions = 1
        self.llm.n = n_completions
        raise NotImplementedError("Propose prompt not yet implemented.")


class ThoughtEvaluator(object):
    METHODS = ('vote', 'value')
    HOW_VOTE = ('majority', 'llm')
    HOW_VALUE = ('llm',)

    def __init__(self, method: str, how: str):
        method, how = method.lower(), how.lower()
        if method not in self.METHODS: raise ValueError(f"method must be either {', '.join(self.METHODS)}")
        if method == 'vote' and how not in self.HOW_VOTE:
            raise ValueError(f"For method=vote, how must be either {', '.join(self.HOW_VOTE)}")
        if method == 'value' and how not in self.HOW_VALUE:
            raise ValueError(f"For method=vote, how must be either {', '.join(self.HOW_VALUE)}")
        self.method = method
        self.how = how

    def evaluate(self, thoughts: list[Thought]):
        if self.method == 'vote':
            # todo: vote across the thoughts (ToT, all thoughts are wrapped as input to the LLM, and got the LLM to vote)
            if self.how == 'majority':
                pass  # requires answer?
            if self.how == 'llm':
                pass  # format each thought as a single Input to the LLM. Ask LLM to vote.

        elif self.method == 'value':
            # todo: give value to each thought
            pass
        else:
            raise NotImplementedError("Only methods vote and value are currently implemented.")


class BaseSearch(object):
    pass


class BFS(BaseSearch):
    def __init__(self, best: int, max_depth: int,
                 generator: ThoughtGenerator, evaluator: ThoughtEvaluator):
        self.best = best
        self.max_steps = max_depth

        self.tgen = generator
        self.teval = evaluator

    def search(self, inp: str) -> list[Thought]:
        states = deque()
        states.appendleft(inp)
        logger.debug(f"Start BFS: input {inp} max steps {self.max_steps} best {self.best}")

        best: list[tuple[Thought, Value]] = list()
        for i in range(self.max_steps):
            logger.debug(f"[step: {i}] states {states}")
            best.clear()
            while (thought := states.popleft()) is not None:
                value = self.evaluate(thought)
                best.append(value)
            best = sorted(best, key=lambda tv: tv[1], reverse=True)[:self.best]  # best candidates
            logger.debug(f"[step: {i}] best {best}")
            assert len(best) == self.best, f"Mismatched number of best candidates. Expect {self.best} Got {len(best)}"
            for b in best:
                thought, value = b
                states.appendleft(thought)
        return [tv[0] for tv in best]

    def evaluate(self, thoughts: list[Thought]) -> V:
        self.teval.evaluate(thoughts)

    def generate(self, **kwargs) -> list[Thought]:
        return self.tgen.generate(**kwargs)


BFSConfig = namedtuple('BFSConfig', ['best', 'max_depth'])

BFSCONFIG_SELFCONSISTENCY = BFSConfig(best=1, max_depth=1)


class ToT(object):
    def __init__(self):
        model_kwargs = Paper_Sampling_Scheme.openai()
        temp = model_kwargs.pop('temperature')
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            n=0,
            temperature=temp,
            model_kwargs=model_kwargs,
        )
        cot_examples = [
            create_cot_prompt_example(),
            create_cot_prompt_example(),
            create_cot_prompt_example(),
        ]
        template = create_cot_prompt_template(instructions='', cot_examples=cot_examples)
        generator = ThoughtGenerator(llm, n_thoughts=3, prompt_template=template)
        evaluator = ThoughtEvaluator(method='vote', how='majority')  # todo: complete thought evaluator
        search = BFS(best=1, max_depth=1, generator=generator, evaluator=evaluator)  # note:

        self.search = search

    def run(self, inp: str) -> list[Thought]:
        final: list[Thought] = self.search.search(inp)
        return final
