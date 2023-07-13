""" Iterative Refinement

"iterative-refine approach on top of an IO sample for at most 10 iterations. At each iteration, the LM is conditioned on all previous history to “reflect on your mistakes and generate a refined answer” if the output is incorrect. Note
that it uses groundtruth feedback signals about equation correctness." -> sounds like recursion.

note:
    Modes:
    1. Human in the loop
    2. Pre-defined set of reasoning
"""