#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${REME_ROOT:-}" ]]; then
  echo "REME_ROOT is required. Set it in .env or export it before running this script." >&2
  exit 1
fi

unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy FTP_PROXY ftp_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

export PYTHONPATH="$REME_ROOT:${PYTHONPATH:-}"
export FLOWLLM_EMBEDDING_OMIT_DIMENSIONS="${FLOWLLM_EMBEDDING_OMIT_DIMENSIONS:-1}"

REME_PORT="${REME_PORT:-8003}"
REME_LLM_MODEL="${REME_LLM_MODEL:-${LLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}}"
REME_EMBED_MODEL="${REME_EMBED_MODEL:-${EMBED_MODEL:-${CONMEM_EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}}}"

cd "$REME_ROOT"
exec reme \
  backend=http \
  http.host="${REME_HOST:-127.0.0.1}" \
  http.port="$REME_PORT" \
  llm.default.model_name="$REME_LLM_MODEL" \
  embedding_model.default.model_name="$REME_EMBED_MODEL" \
  vector_store.default.backend=local
