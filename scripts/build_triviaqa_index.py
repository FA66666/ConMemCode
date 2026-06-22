#!/usr/bin/env python3
"""
Build a TriviaQA vector index with a configurable embedding API.

Supported sources:
1. TriviaQA `rc.wikipedia` evidence pages.
2. Local Search-R1 / FlashRAG-style JSONL corpora, such as `wiki-18.jsonl`.

Usage:
    python scripts/build_triviaqa_index.py --output ./data/triviaqa/index.faiss
"""
import argparse
import os
import pickle
import sys
import tempfile
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from common.utils.index_building import chunk_text, stable_hash


DEFAULT_CORPUS_SOURCE = "triviaqa_evidence"


class EmbeddingClient:
    """Call an OpenAI-compatible embedding API such as SiliconFlow."""
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, embed_dim: int = None):
        # Prefer explicit arguments, then environment variables, to match the retrieval service.
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
                **({"dimensions": self.embed_dim} if self.embed_dim else {})
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Handle common API response variants.
        if "data" in data:
            embeddings = [item["embedding"] for item in data["data"]]
        else:
            embeddings = data.get("embeddings", [])
            
        return np.array(embeddings, dtype=np.float32)


def resolve_dense_doc_prefix(model_name: str | None, explicit_prefix: str | None = None) -> str:
    """Resolve document-side prefix for dense retrievers."""
    if explicit_prefix is not None:
        return explicit_prefix
    model_name = (model_name or "").lower()
    if "e5" in model_name:
        return "passage: "
    return ""


def _extract_unique_triviaqa_documents(ds, max_docs: int = 0) -> list[dict]:
    unique_docs: dict[str, dict] = {}

    print("Extracting TriviaQA Wikipedia evidence documents...")
    for split_name in ds.keys():
        split_ds = ds[split_name]
        split_size = len(split_ds)
        num_to_process = min(max_docs, split_size) if max_docs > 0 else split_size
        for example in tqdm(split_ds.select(range(num_to_process)), desc=f"Scanning {split_name}"):
            entity_pages = example.get("entity_pages", {})
            if not entity_pages:
                continue

            filenames = entity_pages.get("filename", [])
            titles = entity_pages.get("title", [])
            contexts = entity_pages.get("wiki_context", [])

            for filename, title, context in zip(filenames, titles, contexts):
                if not context:
                    continue
                source_id = filename or stable_hash(title, context)
                if source_id in unique_docs:
                    continue
                unique_docs[source_id] = {
                    "source_id": source_id,
                    "title": title or "Untitled",
                    "text": context,
                }

    return list(unique_docs.values())


def _extract_corpus_documents(corpus_path: str, max_docs: int = 0) -> list[dict]:
    """Load Search-R1-style JSONL corpus with `id` and `contents` fields."""
    print(f"Loading external corpus: {corpus_path}")
    cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("json", data_files=corpus_path, split="train", cache_dir=cache_dir)
    if max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    documents: list[dict] = []
    for idx, example in enumerate(tqdm(ds, desc="Scanning external corpus")):
        contents = (example.get("contents") or "").strip()
        if not contents:
            continue
        title, sep, text = contents.partition("\n")
        title = title.strip().strip('"') or "Untitled"
        documents.append({
            "source_id": str(example.get("id", idx)),
            "title": title,
            "text": text if sep else contents,
            "contents": contents,
        })
    return documents


def _chunk_triviaqa_documents(documents: list[dict], chunk_chars: int, overlap_chars: int, min_chunk_chars: int) -> list[dict]:
    chunked_documents: list[dict] = []

    for doc in tqdm(documents, desc="Chunking evidence documents"):
        passages = chunk_text(
            doc["text"],
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
        if not passages:
            continue

        total_chunks = len(passages)
        for idx, passage in enumerate(passages):
            chunk_id = f"{doc['source_id']}::chunk{idx}"
            chunked_documents.append({
                "id": chunk_id,
                "source_id": doc["source_id"],
                "title": doc["title"],
                "chunk_index": idx,
                "chunk_count": total_chunks,
                "contents": f"{doc['title']}\n{passage}",
            })

    return chunked_documents


def build_index(
    output_path: str,
    max_docs: int = 0,
    corpus_source: str = DEFAULT_CORPUS_SOURCE,
    corpus_path: str | None = None,
    chunk_chars: int = 900,
    chunk_overlap_chars: int = 120,
    min_chunk_chars: int = 160,
    doc_prefix: str | None = None,
    save_doc_map: bool | None = None,
    embed_url: str = None,
    api_key: str = None,
    model: str = None,
    embed_dim: int = None,
):
    """Build a FAISS index for TriviaQA documents."""
    try:
        import faiss
    except ImportError:
        print("Please install faiss: pip install faiss-cpu or faiss-gpu")
        sys.exit(1)
    
    if corpus_source == "search_r1_corpus":
        if not corpus_path:
            raise ValueError("--corpus-path is required when using search_r1_corpus")
        unique_docs = _extract_corpus_documents(corpus_path, max_docs=max_docs)
        documents = _chunk_triviaqa_documents(
            unique_docs,
            chunk_chars=chunk_chars,
            overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
    else:
        print("Loading TriviaQA dataset...")
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
        unique_docs = _extract_unique_triviaqa_documents(ds, max_docs=max_docs)
        documents = _chunk_triviaqa_documents(
            unique_docs,
            chunk_chars=chunk_chars,
            overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
    
    print(f"Total documents: {len(documents)}")
    if corpus_source == "search_r1_corpus":
        print(f"Source records from external corpus: {len(unique_docs)}")
    else:
        print(f"Unique evidence pages: {len(unique_docs)}")
    if not documents:
        if corpus_source == "search_r1_corpus":
            print("Error: no passages were extracted from the external JSONL corpus")
        else:
            print("Error: no passages were extracted from TriviaQA evidence documents")
        sys.exit(1)
    
    # Fetch embeddings in batches.
    print("Connecting to embedding service...")
    client = EmbeddingClient(embed_url, api_key, model, embed_dim)
    print(f"Using: {client.base_url}, model={client.model}, dim={client.embed_dim}")
    resolved_doc_prefix = resolve_dense_doc_prefix(client.model if model is None else model, doc_prefix)
    if resolved_doc_prefix:
        print(f"Document encoding prefix: {resolved_doc_prefix!r}")
    
    batch_size = 32
    all_embeddings = []
    
    print("Encoding documents...")
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        texts = [f"{resolved_doc_prefix}{d['contents'][:2048]}" for d in batch]
        embeddings = client.embed(texts)
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Build the FAISS index.
    print("Building FAISS index...")
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product similarity.
    
    # L2-normalize vectors for cosine similarity.
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    # Save the index and optional document map.
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, output_path)
    
    if save_doc_map is None:
        save_doc_map = corpus_source != "search_r1_corpus"
    if save_doc_map:
        doc_map_path = output_path.replace(".faiss", "_docs.pkl")
        with open(doc_map_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Document map saved to: {doc_map_path}")
    else:
        print("Skipped _docs.pkl; pass --corpus-path to load the original JSONL corpus during retrieval")

    print(f"Index saved to: {output_path}")
    print(f"Vector dim: {dim}, document count: {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/triviaqa/index.faiss")
    parser.add_argument("--corpus-source", type=str, default=DEFAULT_CORPUS_SOURCE,
                        choices=["triviaqa_evidence", "search_r1_corpus"],
                        help="Index source: TriviaQA evidence or a Search-R1-style JSONL corpus")
    parser.add_argument("--corpus-path", type=str, default=None,
                        help="External JSONL / JSONL.GZ corpus path for --corpus-source search_r1_corpus")
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum examples per split; 0 means no limit")
    parser.add_argument("--chunk-chars", type=int, default=900, help="Target character count for each passage")
    parser.add_argument("--chunk-overlap-chars", type=int, default=120, help="Character overlap between neighboring passages")
    parser.add_argument("--min-chunk-chars", type=int, default=160, help="Minimum character count for retained passages")
    parser.add_argument("--doc-prefix", type=str, default=None,
                        help="Document encoding prefix; inferred by model by default, e5 models use 'passage: '")
    parser.add_argument("--save-doc-map", action=argparse.BooleanOptionalAction, default=None,
                        help="Whether to save _docs.pkl; defaults to disabled for large Search-R1 corpora")
    parser.add_argument("--embed-url", type=str, default=None, help="Embedding API URL")
    parser.add_argument("--api-key", type=str, default=None, help="Embedding API Key")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--embed-dim", type=int, default=None, help="Embedding dimension (defaults to EMBED_DIM)")
    args = parser.parse_args()

    build_index(
        output_path=args.output,
        max_docs=args.max_docs,
        corpus_source=args.corpus_source,
        corpus_path=args.corpus_path,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
        doc_prefix=args.doc_prefix,
        save_doc_map=args.save_doc_map,
        embed_url=args.embed_url,
        api_key=args.api_key,
        model=args.model,
        embed_dim=args.embed_dim,
    )
