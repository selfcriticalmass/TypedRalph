from __future__ import annotations

import re

from config import RetrievalConfig
from embed import EmbeddingCache, cosine_similarity
from schema import FunctionMatch, FunctionSchema


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class SearchEngine:
    def __init__(self, functions: list[FunctionSchema], config: RetrievalConfig):
        self.functions = functions
        self.config = config
        self.cache = EmbeddingCache(config)
        self.semantic_records: dict[str, object] = {}
        self.semantic_error: str | None = None
        self.refresh(functions)

    def refresh(self, functions: list[FunctionSchema]) -> None:
        self.functions = functions
        self.semantic_error = None
        if not self.config.semantic_enabled or not functions:
            self.semantic_records = {}
            return
        try:
            self.semantic_records = self.cache.ensure_registry_embeddings(functions)
        except Exception as exc:
            self.semantic_records = {}
            self.semantic_error = str(exc)

    def list_functions(self) -> list[FunctionMatch]:
        return [
            FunctionMatch(
                function_name=function.name,
                signature_summary=function.signature_summary(),
                docstring_summary=function.docstring_summary(),
                lexical_score=0.0,
                semantic_score=0.0,
                fused_score=0.0,
                tags=function.tags,
            )
            for function in self.functions
        ]

    def search(self, query: str, *, top_k: int | None = None) -> list[FunctionMatch]:
        top_k = top_k or self.config.top_k
        lexical_scores = self._lexical_scores(query)
        semantic_scores = self._semantic_scores(query)
        lexical_ranks = _rank_scores(lexical_scores, self.config.rrf_k)
        semantic_ranks = _rank_scores(semantic_scores, self.config.rrf_k)

        results: list[FunctionMatch] = []
        for function in self.functions:
            fused = lexical_ranks.get(function.name, 0.0) + semantic_ranks.get(
                function.name, 0.0
            )
            results.append(
                FunctionMatch(
                    function_name=function.name,
                    signature_summary=function.signature_summary(),
                    docstring_summary=function.docstring_summary(),
                    lexical_score=lexical_scores.get(function.name, 0.0),
                    semantic_score=semantic_scores.get(function.name, 0.0),
                    fused_score=fused,
                    tags=function.tags,
                )
            )

        results.sort(key=lambda item: item.fused_score, reverse=True)
        return results[:top_k]

    def _lexical_scores(self, query: str) -> dict[str, float]:
        query_tokens = set(_tokenize(query))
        scores: dict[str, float] = {}
        normalized_query = query.lower().strip()
        for function in self.functions:
            corpus = " ".join(
                [
                    function.signature_text(),
                    function.docstring,
                    " ".join(function.tags),
                ]
            ).lower()
            corpus_tokens = set(_tokenize(corpus))
            overlap = len(query_tokens & corpus_tokens)
            base = overlap / max(len(query_tokens), 1)
            name_bonus = (
                0.5
                if normalized_query and normalized_query in function.name.lower()
                else 0.0
            )
            scores[function.name] = base + name_bonus
        return scores

    def _semantic_scores(self, query: str) -> dict[str, float]:
        if not self.semantic_records:
            return {}
        query_vector = self.cache.embed_query(query)
        scores: dict[str, float] = {}
        for function in self.functions:
            record = self.semantic_records.get(function.name)
            if record is None:
                continue
            scores[function.name] = cosine_similarity(query_vector, record.dense_vector)
        return scores


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _rank_scores(raw_scores: dict[str, float], rrf_k: int) -> dict[str, float]:
    ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    return {name: 1.0 / (rrf_k + index + 1) for index, (name, _) in enumerate(ranked)}
