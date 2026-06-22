#!/usr/bin/env python3
"""
Start an OpenAI-compatible LLM API server with vLLM.

Usage:
    python scripts/deploy_local.py
    python scripts/deploy_local.py --model Qwen/Qwen3-4B-Instruct-2507 --port 8100
    python scripts/deploy_local.py --gpu-memory-utilization 0.35

Then set in .env:
    LLM_BASE_URL=http://localhost:8100/v1
    LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
"""
import argparse
import os
import subprocess
import sys


def main():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    parser = argparse.ArgumentParser(description="Deploy LLM server with vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    cmd = [
        "vllm", "serve", args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",
    ]

    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"API:    http://127.0.0.1:{args.port}/v1")
    print(f"Health: http://127.0.0.1:{args.port}/health")
    print("=" * 60)

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
