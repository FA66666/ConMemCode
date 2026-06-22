import gzip
import json
import os
import pickle
import tempfile
from typing import Any


def resolve_query_prefix(model_name: str | None, explicit_prefix: str | None = None) -> str:
    """Resolve query-side prefix for dense retrievers."""
    if explicit_prefix:
        return explicit_prefix
    model_name = (model_name or "").lower()
    if "e5" in model_name:
        return "query: "
    return ""


def normalize_corpus_document(doc: dict[str, Any], fallback_id: int | None = None) -> dict[str, Any]:
    """Normalize Search-R1-style JSONL corpus rows to the local document schema."""
    if "contents" not in doc:
        return doc

    contents = (doc.get("contents") or "").strip()
    title, sep, text = contents.partition("\n")
    title = title.strip().strip('"')
    normalized = dict(doc)
    normalized.setdefault("id", str(doc.get("id", fallback_id if fallback_id is not None else "")))
    normalized.setdefault("title", title or "Untitled")
    normalized.setdefault("text", text if sep else contents)
    normalized["contents"] = contents
    return normalized


class DocumentStore:
    """Load retrieval documents from `_docs.pkl` or a Search-R1-style corpus JSONL."""

    def __init__(self, index_path: str, corpus_path: str | None = None):
        self.mode = "pickle"
        doc_map_path = index_path.replace(".faiss", "_docs.pkl")

        if os.path.exists(doc_map_path):
            try:
                with open(doc_map_path, "rb") as f:
                    self.documents = pickle.load(f)
                self.source_path = doc_map_path
                return
            except Exception:
                # Some externally prepared indexes ship an incompatible *_docs.pkl.
                # Fall back to the explicit JSONL corpus if available.
                if not corpus_path:
                    raise

        if not corpus_path:
            raise FileNotFoundError(
                f"Could not find document mapping file {doc_map_path}, and --corpus-path was not provided."
            )

        self.mode = "jsonl"
        try:
            from datasets import load_dataset

            cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.documents = load_dataset(
                "json",
                data_files=corpus_path,
                split="train",
                cache_dir=cache_dir,
            )
        except Exception:
            open_fn = gzip.open if corpus_path.endswith(".gz") else open
            with open_fn(corpus_path, "rt", encoding="utf-8") as f:
                self.documents = [json.loads(line) for line in f if line.strip()]
            self.mode = "jsonl_list"
        self.source_path = corpus_path

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        row = self.documents[int(idx)]
        return normalize_corpus_document(row, fallback_id=int(idx))
