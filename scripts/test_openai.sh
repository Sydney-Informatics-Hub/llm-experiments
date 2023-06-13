#!/bin/zsh

[[ -z "$OPENAI_API_KEY" ]] && ( echo "Missing OPENAI_API_KEY."; exit 1 )

MODEL="text-davinci-003"
PROMPT="the meaning of life is"
MAX_TOKENS=30
NUM_COMPLETIONS=2
TEMP=2
TOP_P=0.5
LOGPROB=5
DATA="
{
  \"model\": \"${MODEL}\",
  \"prompt\": \"$PROMPT\",
  \"max_tokens\": ${MAX_TOKENS},
  \"temperature\": ${TEMP},
  \"n\": ${NUM_COMPLETIONS},
  \"top_p\": ${TOP_P},
  \"logprobs\": ${LOGPROB}
}"
echo $DATA | jq || exit 1       # ensure json is formatted correctly.

curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d ${DATA}
