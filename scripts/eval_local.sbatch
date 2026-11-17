#!/usr/bin/env bash
# Serve a local (open-weights) model with vLLM on this GPU node, then run an Inspect eval against it.
# The model is addressed as  openai-api/local/<TAG>  (Inspect's OpenAI-compatible provider).
#
#   MODEL=Qwen/Qwen3.5-4B TAG=qwen35_4b \
#   TASK=tasks/demo.py ARGS="-T harness=react -T tools=sandbox -T search=off" \
#     sbatch --account=<your account> scripts/eval_local.sbatch      # submit from the repo root
#
#   TOOL_PARSER / REASONING_PARSER / EXTRA_VLLM override the per-family defaults below.
#   VLLM_BIN (default: vllm on PATH), HUB (HF cache, default ~/.cache/huggingface) and
#   HF_OFFLINE=1 (use only cached weights) adapt it to your cluster; add #SBATCH lines as needed.
#   Any inspect_evals benchmark works as TASK, e.g. TASK=inspect_evals/gpqa_diamond.
#SBATCH --job-name=inspect-local
#SBATCH --partition=sharing,gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/inspect_local_%j.out
set -euo pipefail
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
HUB=${HUB:-$HOME/.cache/huggingface}
MODEL=${MODEL:?set MODEL to a local model dir or HF id}
TAG=${TAG:-$(basename "$MODEL" | sed 's/^models--//; s/--/_/g')}
TASK=${TASK:-tasks/demo.py}
ARGS=${ARGS:-}
MAX_LEN=${MAX_LEN:-32768}
VLLM_BIN=${VLLM_BIN:-vllm}
LOG_DIR=${LOG_DIR:-$ROOT/logs/$TAG}

# Tool-call / reasoning parsers per model family (vLLM needs the right one for tool use).
lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
case "$lower" in
  *qwen3.5*|*qwen3_5*|*qwen3-coder*) TP=qwen3_coder; RP=qwen3 ;;
  *qwen3*)                           TP=hermes;      RP=qwen3 ;;
  *qwen2.5*)                         TP=hermes;      RP= ;;
  *llama-3*|*llama3*)                TP=llama3_json; RP= ;;
  *mistral*)                         TP=mistral;     RP= ;;
  *gemma*)                           TP=pythonic;    RP= ;;
  *granite*)                         TP=granite;     RP= ;;
  *olmo*)                            TP=olmo3;       RP= ;;
  *)                                 TP=hermes;      RP= ;;
esac
TOOL_PARSER=${TOOL_PARSER:-$TP}; REASONING_PARSER=${REASONING_PARSER:-$RP}

export HF_HOME=$HUB HF_HUB_CACHE=$HUB
[[ ${HF_OFFLINE:-0} == 1 ]] && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$ROOT/.cache/vllm TRITON_CACHE_DIR=$ROOT/.cache/triton TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache/torchinductor
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# The cluster sets http(s)_proxy; local server traffic must bypass it (API judges / search still use it).
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"

PORT=$((20000 + SLURM_JOB_ID % 20000))
export LOCAL_BASE_URL=http://127.0.0.1:$PORT/v1 LOCAL_API_KEY=local
mkdir -p "$LOG_DIR" "$ROOT/logs/slurm"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$VLLM_BIN" serve "$MODEL" --served-model-name "$TAG" --port "$PORT" --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} ${EXTRA_VLLM:-} \
  > "$LOG_DIR/vllm_$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 180); do
  curl -sf --noproxy "*" "$LOCAL_BASE_URL/models" > /dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -50 "$LOG_DIR/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done

cd "$ROOT"
# shellcheck disable=SC2086
.venv/bin/inspect eval "$TASK" --model "openai-api/local/$TAG" --log-dir "$LOG_DIR" \
  --max-connections "${MAX_CONNECTIONS:-32}" $ARGS
