#!/usr/bin/env bash
# Local deployment of Qwen3.5-397B-A17B with vLLM (OpenAI-compatible server on :8000).
# Requires vLLM >= 0.17.0 and 8 data-center GPUs (recipe verified on 8x H200 / 8x MI300X).
#   uv venv && source .venv/bin/activate && uv pip install -U vllm --torch-backend auto
# HALTER only needs text: --language-model-only skips the vision encoder.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}     # or Qwen/Qwen3.5-397B-A17B (BF16), Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
TP=${TP:-8}
PORT=${PORT:-8000}
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port "$PORT" \
  --trust-remote-code
# Debug on one GPU with a smaller model, e.g.:
#   MODEL=Qwen/Qwen3-8B TP=1 bash scripts/serve_qwen_vllm.sh   (drop --language-model-only for non-VL models)
