#!/bin/bash
# Thin wrapper: use MPO's vLLM launcher (OpenAI-compatible API for Qwen2.5-VL).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/MPO/servers/vllm_server/vllm_activate.sh"
