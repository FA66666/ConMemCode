#!/usr/bin/env bash
# Source this file to configure the current shell for ConMem experiments.
#
# Usage: source scripts/activate_conmem_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'This script must be sourced, not executed: source %s\n' "$0" >&2
  exit 1
fi

_conmem_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PROJECT_ROOT="$(cd -- "$_conmem_script_dir/.." >/dev/null 2>&1 && pwd)"
unset _conmem_script_dir

# Activate conda environment
if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate conmem || return 1
cd "$PROJECT_ROOT" || return 1

# Add project root to PYTHONPATH
case ":${PYTHONPATH:-}:" in
  *":$PROJECT_ROOT:"*) ;;
  *) export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# Convenience variables
export CONFIG_ROOT="$PROJECT_ROOT/configs/conmem"
export CONMEM_SHARED_STORAGE="$PROJECT_ROOT/conmem_shared_storage"

# Service ports and model names
export LLM_PORT=8100
export EMBED_PORT=8002
export SEARCH_PORT=8000

export LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"

export LLM_BASE_URL="http://127.0.0.1:$LLM_PORT/v1"
export EMBED_BASE_URL="http://127.0.0.1:$EMBED_PORT/v1"
export SEARCH_URL="http://127.0.0.1:$SEARCH_PORT/retrieve"

export COMMON_API_ARGS="--api_base $LLM_BASE_URL"
export QA_API_ARGS="--api_base $LLM_BASE_URL --search_url $SEARCH_URL"

# Data paths
export DATA_ROOT="$PROJECT_ROOT/data"
export TRIVIAQA_INDEX="$DATA_ROOT/triviaqa/index.faiss"
export POPQA_INDEX="$DATA_ROOT/popqa/index.faiss"
