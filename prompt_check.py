""" Prompt Checks
A collection of general functions to validate your prompt.

Example Usage:
prompt = PromptTemplate(
        input_variables=['query'],
        template=raw_template,
    ).format("your query here ")
passed, reasons = prompt_check(prompt)
assert passed, reasons
"""
import sys

PASSED = bool


def prompt_check(prompt: str) -> str:
    if not isinstance(prompt, str): raise ValueError("prompt must be a string.")
    checks = dict()
    for check, fn in checks.items():
        passed, reason = fn(prompt)
        if not passed:
            checks[check] = reason
    if len(checks) > 0:
        print(checks, file=sys.stderr)
        raise ValueError("prompt did not pass the checks.")
    return prompt


def _trailing_whitespace(prompt: str) -> tuple[PASSED, str]:
    # https://twitter.com/karpathy/status/1657949234535211009
    openai_warning = "Warning: Your text ends in a trailing space, " \
                     "which causes worse performance due to how the API splits text into tokens."
    reason = "Including a trailing whitespace will cause the tokeniser to encode the word into another token ID " \
             "that's possibly rare. Hence degrade performance of the model."
    return not prompt.endswith(' '), reason


checks = {
    _trailing_whitespace.__name__: _trailing_whitespace
}

__all__ = ['prompt_check']
