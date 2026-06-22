#!/usr/bin/env python3
"""
Start an OpenAI-compatible embedding API server with vLLM.

Usage:
    python scripts/deploy_embedding.py
    python scripts/deploy_embedding.py --model Qwen/Qwen3-Embedding-0.6B --port 8002
    python scripts/deploy_embedding.py --gpu-memory-utilization 0.05
"""
import argparse
import os
import subprocess
import sys


def main():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    parser = argparse.ArgumentParser(description="Deploy embedding server with vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    args = parser.parse_args()

    cmd = [
        "vllm", "serve", args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",
    ]

    print(f"Embedding model: {args.model}")
    print(f"API:    http://127.0.0.1:{args.port}/v1")
    print(f"Health: http://127.0.0.1:{args.port}/health")
    print(f"GPU memory utilization: {args.gpu_memory_utilization:.0%}")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
