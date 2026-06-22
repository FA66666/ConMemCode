# ConMem

<p align="center">
  <strong>Trajectory-to-Memory Conditioning for Multi-Agent Systems</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Executors" src="https://img.shields.io/badge/Executors-Qwen-purple">
  <img alt="MAS" src="https://img.shields.io/badge/MAS-CAMEL%20%7C%20MacNet%20%7C%20AutoGen-lightgrey">
</p>

ConMem is a pluggable memory layer for multi-agent systems that converts
completed task trajectories into structured, typed memory cards and injects the
most useful cards back into later agent prompts.

After each task, the trajectory goes through three stages:

- **Interpret** — compress the trajectory into candidate memory cards (state / plan / exec / eval).
- **Admit** — score cards on reliability, novelty, recency, and utility; drop low-quality ones.
- **Retrieve & coordinate** — trigger relevant cards by semantic match, walk the memory graph, merge duplicates, resolve conflicts, and serialize into budgeted hints for the agent prompt.

The memory graph evolves across tasks: new cards link to existing ones via relation edges,
and a post-commit merge pass consolidates near-duplicates. Plugs into CAMEL, MacNet,
AutoGen, and single-agent hosts through `on_task_start` / `on_task_complete`.

This directory is a curated open-source snapshot. It intentionally excludes large
model checkpoints, raw private server paths, and transient experiment ledgers.

## Quick Links

| item | path |
|---|---|
| Environment activation | `source scripts/activate_conmem_env.sh` |
| LLM launcher | `scripts/deploy_local.py` |
| Embedding launcher | `scripts/deploy_embedding.py` |
| Unified benchmark runner | `scripts/run_benchmarks.py` |
| Memory card viewer | `scripts/view_cards.py` |
| Offline store compaction | `scripts/compact_memory_store.py` |
| Retrieval index builder | `scripts/build_triviaqa_index.py` / `scripts/build_popqa_index.py` |
| Retrieval service | `scripts/deploy_retrieval.py` |
| ReMe HTTP service launcher | `scripts/run_reme_http_service.sh` |

## Layout

```text
conmem/
  mas_core/
    memory/backbone/
      conmem/               # ConMem core: orchestration, interpreter, admission,
                            #   retriever, graph, coordinator, serializer, storage
      simplemem/            # SimpleMem trajectory-retrieval baseline adapter
      reme/                 # ReMe HTTP adapter
    structures/
      camel/                # CAMEL-style role-playing MAS host
      macnet/               # MacNet-style multi-agent MAS host
      autogen/              # AutoGen-style conversational MAS host
  data/
    kodcode/                # KodCode benchmark env and builder
    triviaqa/               # TriviaQA benchmark env and builder
    popqa/                  # PopQA benchmark env, builder, retriever, and sampling
    pddl/                   # PDDL planning benchmark env and builder
  configs/conmem/           # Per-benchmark YAML configuration files
  scripts/                  # Evaluation, indexing, retrieval, and inspection scripts
  common/                   # Shared interaction and utility code
  interactions/             # Interaction manager entry points
  utils/                    # Config loader, message, agent, stats utilities
```

## Installation

ConMem targets Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install optional dependencies for specific benchmarks:

```bash
pip install -e ".[datasets,qa]"   # TriviaQA, PopQA
pip install -e ".[pddl]"          # PDDL planning
pip install -e ".[hf_local]"      # local HF models (torch, transformers, accelerate)
pip install -e ".[local_model]"   # local vLLM serving
```

To reproduce the exact environment, install the full locked dependency set:

```bash
pip install -r requirements.txt
```

## Environment

Create a local `.env` from the template:

```bash
cp .env.example .env
```

Key variables:

| variable | purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible chat endpoint |
| `CONMEM_EMBED_BASE_URL`, `CONMEM_EMBED_API_KEY`, `CONMEM_EMBED_MODEL` | embedding endpoint for ConMem retrieval |
| `SEARCH_URL` | optional retrieval endpoint for factual QA environments |
| `TOPK` | retrieval count (default: 5) |
| `CONMEM_TOKEN_BUDGET` | prompt budget in tokens (default: 1536) |

## Data Preparation

QA benchmarks (TriviaQA / PopQA) need a Wikipedia retrieval index before running:

```bash
python scripts/build_triviaqa_index.py
python scripts/build_popqa_index.py
```

## Start Services

Each new terminal:

```bash
source scripts/activate_conmem_env.sh
```

### Terminal 1 — LLM

```bash
python scripts/deploy_local.py
```

Or manually:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8100 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager
```

### Terminal 2 — Embedding

```bash
python scripts/deploy_embedding.py
```

Or manually:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 2048 --gpu-memory-utilization 0.05 \
  --enforce-eager
```

### Terminal 3 — Retrieval (QA benchmarks only)

```bash
python scripts/deploy_retrieval.py --index data/triviaqa/index.faiss --port 8000
```

### Health check

```bash
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8000/health
```

## Quick Start

```bash
source scripts/activate_conmem_env.sh
```

### Smoke test

```bash
python scripts/run_benchmarks.py \
  --benchmark kodcode \
  --mas_type macnet \
  --memory_backend conmem \
  --data_split test \
  --num_tasks 20
```

### Supported benchmarks

| benchmark | domain | metric | tasks | command |
|---|---|---|---|---|
| `kodcode` | Python code generation | pass rate | 2 000 | `$COMMON_API_ARGS` |
| `triviaqa` | factual QA + retrieval | answer match | 6 993 | `$QA_API_ARGS` |
| `popqa` | factual QA + retrieval | answer match | 7 267 | `$QA_API_ARGS` |
| `pddl` | symbolic planning | win rate | 60 | `$COMMON_API_ARGS` |

Substitute `<benchmark>`, `<mas_type>`, and `<api_args>` into the pattern below:

```bash
python scripts/run_benchmarks.py \
  <api_args> \
  --benchmark <benchmark> \
  --mas_type <mas_type> \
  --data_split test \
  --num_tasks <tasks> \
  --storage_dir results/phase1/<mas_type> \
  --memory_storage_dir banks/<mas_type>_bank
```

Examples:

```bash
# KodCode × AutoGen
python scripts/run_benchmarks.py $COMMON_API_ARGS \
  --benchmark kodcode --mas_type autogen --num_tasks 2000

# TriviaQA × AutoGen
python scripts/run_benchmarks.py $QA_API_ARGS \
  --benchmark triviaqa --mas_type autogen --num_tasks 6993

# All benchmarks (small run)
python scripts/run_benchmarks.py $COMMON_API_ARGS \
  --benchmark all --mas_type autogen --num_tasks 50 --run_baseline
```

### MAS modes

`--mas_type` supports four modes. Commands differ only in that flag:

| mode | description |
|---|---|
| `single` | Single LLM agent |
| `autogen` | AutoGen-style conversational MAS |
| `camel` | CAMEL-style role-playing MAS |
| `macnet` | MacNet-style multi-agent MAS |

Each mode writes to a separate card bank (`--memory_storage_dir banks/<mode>_bank`).

### Other memory backends

```bash
# SimpleMem baseline
python scripts/run_benchmarks.py $COMMON_API_ARGS \
  --benchmark kodcode --memory_backend simplemem --num_tasks 200

# ReMe (requires ReMe HTTP service)
bash scripts/run_reme_http_service.sh
python scripts/run_benchmarks.py $COMMON_API_ARGS \
  --benchmark kodcode --memory_backend reme --num_tasks 200
```

## Ablation

Toggle ConMem components via environment variables. Each ablation needs its own `--memory_storage_dir`.

| ablation | flag |
|---|---|
| No graph expansion | `CONMEM_ENABLE_GRAPH_EXPANSION=0` |
| No coordination | `CONMEM_ENABLE_COORDINATION=0` |
| No failure reflection | `CONMEM_ENABLE_FAILURE_REFLECTION=0` |
| No failure admission | `CONMEM_ENABLE_FAILURE_ADMISSION=0` |

```bash
CONMEM_ENABLE_GRAPH_EXPANSION=0 \
python scripts/run_benchmarks.py $COMMON_API_ARGS \
  --benchmark kodcode --mas_type autogen --num_tasks 1000 \
  --storage_dir results/ablation/no_graph/autogen \
  --memory_storage_dir results/ablation/no_graph/autogen/kodcode_bank
```

Substitute `ablation`, `benchmark`, `mas_type`, `num_tasks` as needed.

## Parallel Runs

Runs with different `--memory_storage_dir` can execute in parallel:

```bash
python scripts/run_benchmarks.py --benchmark kodcode --mas_type autogen \
  --memory_storage_dir banks/autogen_bank > logs/autogen_kodcode.log 2>&1 &

python scripts/run_benchmarks.py --benchmark kodcode --mas_type camel \
  --memory_storage_dir banks/camel_bank > logs/camel_kodcode.log 2>&1 &

wait
```

## Inspecting Memory

```bash
python scripts/view_cards.py --storage conmem_shared_storage --stats
python scripts/view_cards.py --storage conmem_shared_storage --domain kodcode --limit 10
python scripts/view_cards.py --storage conmem_shared_storage --graph --domain kodcode
python scripts/view_cards.py --storage conmem_shared_storage --card <card_id_prefix> --full
```

## Notes

- Raw benchmark datasets are loaded from HuggingFace at runtime and are not duplicated here.
- Full model checkpoints are not included. Configs and `.env` point to local vLLM or API endpoints.
- `scripts/view_cards.py` reads the SQLite store directly with Python's standard library — no runtime dependencies needed for inspection.

## Citation

Citation information will be added with the public paper release.
