#!/usr/bin/env bash
set -euo pipefail

MODEL_CKPT_PATH=${1:?MODEL_CKPT_PATH is required}
MODEL_TYPE=${2:-"auto"}
num_fs=${3:-"0"}
OUTPUT_PATH=${4:-""}
TASKS=${5:-"arc_challenge,arc_easy,openbookqa,hellaswag,piqa,winogrande,social_iqa,sciq"}
TOKENIZER_PATH=${6:-"auto"}
EXTRA_LM_EVAL_ARGS=()
if [[ "${TOKENIZER_PATH}" == --* ]]; then
  TOKENIZER_PATH="auto"
  EXTRA_LM_EVAL_ARGS=("${@:6}")
elif [[ $# -gt 6 ]]; then
  EXTRA_LM_EVAL_ARGS=("${@:7}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOGPT_NEXT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NANOGPT_NEXT_REPO_ROOT

if [[ "${MODEL_TYPE}" == "auto" ]]; then
  unset NANOGPT_NEXT_MODEL_TYPE
else
  export NANOGPT_NEXT_MODEL_TYPE="${MODEL_TYPE}"
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  if [[ "${MODEL_TYPE}" == "auto" ]]; then
    OUTPUT_PATH="results/$(basename "${MODEL_CKPT_PATH}")"
  else
    OUTPUT_PATH="results/${MODEL_TYPE}"
  fi
fi

MODEL_ARGS="pretrained=${MODEL_CKPT_PATH}"
if [[ "${TOKENIZER_PATH}" != "auto" ]]; then
  MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER_PATH}"
fi

lm_eval \
  --model hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}" \
  --num_fewshot "${num_fs}" \
  "${EXTRA_LM_EVAL_ARGS[@]}"
