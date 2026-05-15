from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from pydantic import BaseModel, Field

from config import RetrievalConfig
from schema import FunctionSchema, registry_schema_hash


class EmbeddingRecord(BaseModel):
    function_name: str
    signature_text: str
    docstring: str
    dense_vector: list[float] = Field(default_factory=list)
    sparse_vector: dict[str, float] = Field(default_factory=dict)
    schema_hash: str
    updated_at: str


class BgeM3Embedder:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from FlagEmbedding import BGEM3FlagModel

        # Force a single target device so FlagEmbedding does not switch into
        # multiprocessing mode across all visible GPUs. That multiprocessing
        # path is what causes the brittle fd/spawn errors inside the TUI.
        self._model = BGEM3FlagModel(
            self.config.embedding_model,
            use_fp16=self.config.use_fp16,
            devices=self._target_device(),
        )
        return self._model

    def _target_device(self) -> str:
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def embed_texts(
        self, texts: list[str]
    ) -> list[tuple[list[float], dict[str, float]]]:
        if not texts:
            return []
        model = self._ensure_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except TypeError:
            encoded = model.encode(
                texts,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
            )

        dense_vectors = encoded.get("dense_vecs")
        if dense_vectors is None:
            dense_vectors = encoded.get("dense_embeddings")
        if dense_vectors is None:
            dense_vectors = []

        sparse_vectors = encoded.get("lexical_weights")
        if sparse_vectors is None:
            sparse_vectors = [{} for _ in texts]
        bundles: list[tuple[list[float], dict[str, float]]] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors):
            dense_list = dense.tolist() if hasattr(dense, "tolist") else list(dense)
            sparse_map = (
                {str(key): float(value) for key, value in dict(sparse).items()}
                if sparse
                else {}
            )
            bundles.append((dense_list, sparse_map))
        return bundles

    def embed_query(self, query: str) -> list[float]:
        dense, _ = self.embed_texts([query])[0]
        return dense


class EmbeddingCache:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.cache_path = Path(config.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = BgeM3Embedder(config)

    def ensure_registry_embeddings(
        self, functions: list[FunctionSchema]
    ) -> dict[str, EmbeddingRecord]:
        schema_hash = registry_schema_hash(functions)
        cached = self._load_cache(schema_hash)
        if cached is not None and len(cached) == len(functions):
            return {record.function_name: record for record in cached}

        texts = [
            function.docstring or function.signature_text() for function in functions
        ]
        embeddings = self.embedder.embed_texts(texts)
        timestamp = datetime.now(timezone.utc).isoformat()
        records = [
            EmbeddingRecord(
                function_name=function.name,
                signature_text=function.signature_text(),
                docstring=function.docstring,
                dense_vector=dense,
                sparse_vector=sparse,
                schema_hash=schema_hash,
                updated_at=timestamp,
            )
            for function, (dense, sparse) in zip(functions, embeddings)
        ]
        self._write_cache(records)
        return {record.function_name: record for record in records}

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)

    def _load_cache(self, schema_hash: str) -> list[EmbeddingRecord] | None:
        if not self.cache_path.exists():
            return None
        frame = pd.read_parquet(self.cache_path)
        if frame.empty:
            return None
        if set(frame["schema_hash"].unique()) != {schema_hash}:
            return None
        records = []
        for row in frame.to_dict(orient="records"):
            records.append(
                EmbeddingRecord(
                    function_name=row["function_name"],
                    signature_text=row["signature_text"],
                    docstring=row["docstring"],
                    dense_vector=json.loads(row["dense_vector"]),
                    sparse_vector=json.loads(row["sparse_vector"]),
                    schema_hash=row["schema_hash"],
                    updated_at=row["updated_at"],
                )
            )
        return records

    def _write_cache(self, records: list[EmbeddingRecord]) -> None:
        frame = pd.DataFrame(
            [
                {
                    "function_name": record.function_name,
                    "signature_text": record.signature_text,
                    "docstring": record.docstring,
                    "dense_vector": json.dumps(record.dense_vector),
                    "sparse_vector": json.dumps(record.sparse_vector),
                    "schema_hash": record.schema_hash,
                    "updated_at": record.updated_at,
                }
                for record in records
            ]
        )
        frame.to_parquet(self.cache_path, index=False)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
