#!/usr/bin/env python3
"""
Build a PopQA vector index with an external embedding API such as SiliconFlow.

By default, this script streams pages from a fixed Wikipedia snapshot and chunks
them into passages. It intentionally does not build an oracle index from PopQA
subject/object Wikipedia titles, because that metadata is not visible in a real
open-domain evaluation setting.

Usage:
    python scripts/build_popqa_index.py --output ./data/popqa/index.faiss

Custom embedding API:
    python scripts/build_popqa_index.py --output ./data/popqa/index.faiss \
        --embed-url https://api.siliconflow.cn/v1 \
        --api-key your-api-key \
        --model intfloat/e5-base-v2
"""
import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from common.utils.index_building import chunk_text, stable_hash


DEFAULT_WIKI_CONFIG = "20231101.en"
AUTO_WIKI_CONFIG = "auto"
DEFAULT_INDEX_SOURCE = "full_wikipedia"


def resolve_dense_doc_prefix(model_name: str | None, explicit_prefix: str | None = None) -> str:
    if explicit_prefix is not None:
        return explicit_prefix
    model_name = (model_name or "").lower()
    if "e5" in model_name:
        return "passage: "
    return ""


class EmbeddingClient:
    """Call an external embedding API such as SiliconFlow."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        embed_dim: int = None,
    ):
        self.base_url = base_url or os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
        self.api_key = api_key or os.getenv("EMBED_API_KEY")
        self.model = model or os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
        self.embed_dim = embed_dim or int(os.getenv("EMBED_DIM", "0")) or None

        import requests

        self.requests = requests

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return embedding vectors for a batch of texts."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={
                "input": texts,
                "model": self.model,
                **({"dimensions": self.embed_dim} if self.embed_dim else {}),
            },
        )
        response.raise_for_status()
        data = response.json()

        if "data" in data:
            embeddings = [item["embedding"] for item in data["data"]]
        else:
            embeddings = data.get("embeddings", [])

        return np.array(embeddings, dtype=np.float32)


def resolve_wikipedia_config(
    requested_config: str = DEFAULT_WIKI_CONFIG,
    available_configs: List[str] | None = None,
) -> str:
    """Select an available English Wikipedia snapshot."""
    if available_configs is None:
        available_configs = list(get_dataset_config_names("wikimedia/wikipedia"))

    english_configs = sorted([cfg for cfg in available_configs if cfg.endswith(".en")], reverse=True)
    latest_english = english_configs[0] if english_configs else None

    if requested_config == AUTO_WIKI_CONFIG:
        if latest_english:
            print(f"Auto-selected Wikipedia config: {latest_english}")
            return latest_english
        raise RuntimeError("No available English Wikipedia config was found")

    if requested_config in available_configs:
        return requested_config

    raise RuntimeError(
        f"Requested Wikipedia config {requested_config!r} is unavailable; "
        f"the latest available English config is {latest_english!r}. "
        f"Use --wiki-config {AUTO_WIKI_CONFIG} to auto-select a config."
    )


def _normalize_title(title: str | None) -> str:
    return (title or "").replace("_", " ").strip()


def load_wikipedia_documents(
    max_docs: int = 0,
    wiki_config: str = DEFAULT_WIKI_CONFIG,
    chunk_chars: int = 900,
    chunk_overlap_chars: int = 120,
    min_chunk_chars: int = 160,
    index_source: str = DEFAULT_INDEX_SOURCE,
) -> list[dict]:
    """
    Load Wikipedia documents as the PopQA retrieval source.

    - `full_wikipedia`: stream Wikipedia pages in order; `max_docs` caps page count.
    """
    if index_source != "full_wikipedia":
        raise ValueError("PopQA indexing no longer supports dataset-metadata entity filters")

    try:
        print("Loading Wikipedia dataset...")
        config_name = resolve_wikipedia_config(wiki_config)
        wiki_ds = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load Wikipedia dataset: {e}")
        print("Ensure Hugging Face is reachable or provide local data")
        sys.exit(1)

    documents: list[dict] = []
    matched_pages = 0
    progress_total = max_docs if max_docs > 0 else None

    for example in tqdm(wiki_ds, desc="Loading documents", total=progress_total):
        if max_docs > 0 and matched_pages >= max_docs:
            break

        title = _normalize_title(example.get("title", ""))
        text = example.get("text", "")

        if not text:
            continue

        passages = chunk_text(
            text,
            chunk_chars=chunk_chars,
            overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
        if not passages:
            continue

        source_id = f"wiki_{stable_hash(title, text)}"
        matched_pages += 1

        for chunk_idx, passage in enumerate(passages):
            documents.append(
                {
                    "id": f"{source_id}::chunk{chunk_idx}",
                    "source_id": source_id,
                    "title": title,
                    "chunk_index": chunk_idx,
                    "chunk_count": len(passages),
                    "contents": f"{title}\n{passage}",
                }
            )

    print(f"Extracted {matched_pages} Wikipedia pages into {len(documents)} passages")
    return documents


def build_index(
    output_path: str,
    max_docs: int = 0,
    wiki_config: str = DEFAULT_WIKI_CONFIG,
    chunk_chars: int = 900,
    chunk_overlap_chars: int = 120,
    min_chunk_chars: int = 160,
    index_source: str = DEFAULT_INDEX_SOURCE,
    doc_prefix: str | None = None,
    embed_url: str = None,
    api_key: str = None,
    model: str = None,
    embed_dim: int = None,
):
    """Build a FAISS index for PopQA."""
    try:
        import faiss
    except ImportError:
        print("Please install faiss: pip install faiss-cpu or faiss-gpu")
        sys.exit(1)

    documents = load_wikipedia_documents(
        max_docs=max_docs,
        wiki_config=wiki_config,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        min_chunk_chars=min_chunk_chars,
        index_source=index_source,
    )

    if not documents:
        print("Error: no documents were loaded")
        sys.exit(1)

    print(f"Total passages: {len(documents)}")

    print(f"Connecting to embedding service: {embed_url or os.getenv('EMBED_BASE_URL', 'https://api.siliconflow.cn/v1')}")
    client = EmbeddingClient(embed_url, api_key, model, embed_dim)
    print(f"Using: {client.base_url}, model={client.model}, dim={client.embed_dim}")
    resolved_doc_prefix = resolve_dense_doc_prefix(client.model if model is None else model, doc_prefix)
    if resolved_doc_prefix:
        print(f"Document encoding prefix: {resolved_doc_prefix!r}")

    batch_size = 32
    all_embeddings = []

    print("Encoding documents...")
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        texts = [f"{resolved_doc_prefix}{d['contents'][:2048]}" for d in batch]
        try:
            embeddings = client.embed(texts)
            all_embeddings.append(embeddings)
        except Exception as e:
            raise RuntimeError(
                f"Encoding batch {i // batch_size} failed; stopped before writing a corrupted index: {e}"
            ) from e

    all_embeddings = np.vstack(all_embeddings)

    print("Building FAISS index...")
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, output_path)

    doc_map_path = output_path.replace(".faiss", "_docs.pkl")
    with open(doc_map_path, "wb") as f:
        pickle.dump(documents, f)

    print(f"Index saved to: {output_path}")
    print(f"Document map saved to: {doc_map_path}")
    print(f"Vector dim: {dim}, document count: {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a PopQA vector index")
    parser.add_argument("--output", type=str, default="./data/popqa/index.faiss", help="Output FAISS index path")
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum Wikipedia pages/articles to keep; 0 means no limit")
    parser.add_argument(
        "--wiki-config",
        type=str,
        default=DEFAULT_WIKI_CONFIG,
        help="Wikipedia dataset config. Use a fixed version for experiments, such as 20231101.en; use auto to select the latest English snapshot",
    )
    parser.add_argument(
        "--index-source",
        type=str,
        default=DEFAULT_INDEX_SOURCE,
        choices=["full_wikipedia"],
        help="Index source. PopQA no longer supports oracle entity filtering from dataset metadata",
    )
    parser.add_argument("--chunk-chars", type=int, default=900, help="Target character count for each passage")
    parser.add_argument("--chunk-overlap-chars", type=int, default=120, help="Character overlap between neighboring passages")
    parser.add_argument("--min-chunk-chars", type=int, default=160, help="Minimum character count for retained passages")
    parser.add_argument("--doc-prefix", type=str, default=None,
                        help="Document encoding prefix; inferred by model by default, e5 models use 'passage: '")
    parser.add_argument("--embed-url", type=str, default=None, help="Embedding API URL (defaults to EMBED_BASE_URL)")
    parser.add_argument("--api-key", type=str, default=None, help="Embedding API key (defaults to EMBED_API_KEY)")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name (defaults to EMBED_MODEL)")
    parser.add_argument("--embed-dim", type=int, default=None, help="Embedding dimension (defaults to EMBED_DIM)")
    args = parser.parse_args()

    build_index(
        output_path=args.output,
        max_docs=args.max_docs,
        wiki_config=args.wiki_config,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
        index_source=args.index_source,
        doc_prefix=args.doc_prefix,
        embed_url=args.embed_url,
        api_key=args.api_key,
        model=args.model,
        embed_dim=args.embed_dim,
    )
