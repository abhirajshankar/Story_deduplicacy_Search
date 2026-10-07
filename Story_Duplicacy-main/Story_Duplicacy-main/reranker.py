# reranker.py
# ─────────────────────────────────────────────────────────────────────
# Cross-encoder reranking, per-client thresholds.
# Returns ALL candidates above rerank_threshold — not just the top one.
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass

from config import CLIENT_CONFIGS, USE_RERANKER
from store import Candidate

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    candidate:    Candidate
    rerank_score: float


class Reranker:

    def __init__(self):
        self._models = {}   # model_name -> loaded reranker instance
        if USE_RERANKER:
            self._load_all_rerankers()

    def _load_all_rerankers(self) -> None:
        unique_models = {cfg["reranker_model"] for cfg in CLIENT_CONFIGS.values()}
        for model_name in unique_models:
            logger.info("Loading reranker: %s", model_name)
            self._models[model_name] = self._load_one(model_name)

    @staticmethod
    def _load_one(model_name: str):
        try:
            from FlagEmbedding import FlagReranker
            return ("flagembedding", FlagReranker(model_name, use_fp16=True))
        except ImportError:
            from sentence_transformers import CrossEncoder
            logger.warning("FlagEmbedding not installed — using CrossEncoder fallback for %s", model_name)
            return ("cross_encoder", CrossEncoder(model_name))

    def rerank(self, client_name: str, query_text: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        """
        Returns ALL candidates whose rerank score clears this client's rerank_threshold,
        sorted descending. If USE_RERANKER is False, falls back to fused_score directly.
        """
        if not candidates:
            return []

        client_cfg = CLIENT_CONFIGS[client_name]
        threshold  = client_cfg["rerank_threshold"]

        if not USE_RERANKER:
            ranked = [
                RankedCandidate(candidate=c, rerank_score=c.fused_score)
                for c in candidates if c.fused_score >= threshold
            ]
            ranked.sort(key=lambda r: r.rerank_score, reverse=True)
            return ranked

        backend, model = self._models[client_cfg["reranker_model"]]
        pairs = [(query_text, f"{c.story.title}. {c.story.body}") for c in candidates]

        if backend == "flagembedding":
            scores = model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
        else:
            scores = model.predict(pairs)

        ranked: list[RankedCandidate] = []
        for cand, score in zip(candidates, scores):
            float_score = float(score)
            if float_score >= threshold:
                ranked.append(RankedCandidate(candidate=cand, rerank_score=float_score))

        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        logger.info("[%s] %d / %d candidates passed rerank threshold %.2f",
                     client_name, len(ranked), len(candidates), threshold)
        return ranked
