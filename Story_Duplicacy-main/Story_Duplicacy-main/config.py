# config.py
# ─────────────────────────────────────────────────────────────────────
# Multi-publication, multi-language news event matcher.
#
# ONE shared Qdrant collection. Matching never crosses publications —
# client_name is a hard filter on every search. Each client has its own
# embedding model set, fusion weights, and thresholds, calibrated per
# language since cosine similarity scores are NOT comparable across
# different models/languages.
# ─────────────────────────────────────────────────────────────────────

# ── Qdrant ────────────────────────────────────────────────────────────
QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "news_stories"

# ── Retention ─────────────────────────────────────────────────────────
RETENTION_DAYS = 3     # stories older than this are hard-deleted by the cleanup job
TIME_WINDOW_DAYS = 1   # matching only considers stories within this window
                        # (kept as a separate constant in case you ever want
                        # retention and the matching window to diverge)

# ── Reranker ──────────────────────────────────────────────────────────
# Shared across clients by default. Override per-client below if you find
# a better model for a specific language later.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = False

# ── Per-client configuration ────────────────────────────────────────────
# Each client is a SELF-CONTAINED matching universe: its own models,
# fusion weights, and thresholds. Add new clients here.
#
# IMPORTANT: client_name strings here are the ONLY valid values accepted
# by the API — unknown client names are rejected (see api.py), to avoid
# typos silently creating orphaned matching universes.
#
# models: list of (vector_name, model_name) — vector_name must be globally
#         unique across the WHOLE config (Qdrant vector names live in one
#         shared collection schema), so prefix with client or language if
#         there's any chance of collision.
# weights: fusion weights for this client's models — must sum to 1.0
# similarity_threshold: pre-rerank cosine gate — START HERE, recalibrate
#         with real labeled pairs per client.
# rerank_threshold: final decision gate after reranking — same caveat.

CLIENT_CONFIGS = {
    "PTI": {
        "language": "en",
        "models": [
            ("vec_pti_mpnet",  "all-mpnet-base-v2"),
            ("vec_pti_minilm", "all-MiniLM-L6-v2"),
        ],
        "weights": {
            "vec_pti_mpnet":  0.65,
            "vec_pti_minilm": 0.35,
        },
        "similarity_threshold": 0.80,   # English models are well-trained — can afford a tighter gate
        "rerank_threshold":     0.40,
        "reranker_model":       DEFAULT_RERANKER_MODEL,
    },

    "Eenadu": {
        "language": "te",
        "models": [
            ("vec_eenadu_bge_m3",       "BAAI/bge-m3"),
            ("vec_eenadu_telugu_sbert", "l3cube-pune/telugu-sentence-bert-nli"),
            ("vec_eenadu_mulitlingual_e5", "intfloat/multilingual-e5-large"),
            # ("vec_eenadu_multilingual_mpnet_base_v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            # ("vec_eenadu_multilingual_MiniLM_L12_v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        ],
        "weights": {
            "vec_eenadu_bge_m3":       0.40,
            "vec_eenadu_telugu_sbert": 0.40,
            "vec_eenadu_mulitlingual_e5": 0.20,
            # "vec_eenadu_multilingual_mpnet_base_v2": 0.20,
            # "vec_eenadu_multilingual_MiniLM_L12_v2": 0.20,
        },
        "similarity_threshold": 0.70,   # Telugu-direct embeddings score lower on average — wider net
        "rerank_threshold":     0.30,
        "reranker_model":       DEFAULT_RERANKER_MODEL,
    },

    "Dainik Jagran": {
        "language": "hi",
        "models": [
            ("vec_jagran_bge_m3",       "BAAI/bge-m3"),
            ("vec_jagran_mpnet_multi",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        ],
        "weights": {
            "vec_jagran_bge_m3":      0.65,
            "vec_jagran_mpnet_multi": 0.35,
        },
        "similarity_threshold": 0.72,
        "rerank_threshold":     0.32,
        "reranker_model":       DEFAULT_RERANKER_MODEL,
    },

    "Lokmat": {
        "language": "mr",
        "models": [
            ("vec_lokmat_bge_m3",      "BAAI/bge-m3"),
            ("vec_lokmat_mpnet_multi", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        ],
        "weights": {
            "vec_lokmat_bge_m3":      0.65,
            "vec_lokmat_mpnet_multi": 0.35,
        },
        "similarity_threshold": 0.72,
        "rerank_threshold":     0.32,
        "reranker_model":       DEFAULT_RERANKER_MODEL,
    },
}

# Vector dimensions — MUST match each model's actual output dimension.
# Shared lookup table since multiple clients may reuse the same underlying model.
MODEL_DIMS = {
    "all-mpnet-base-v2":                                              768,
    "all-MiniLM-L6-v2":                                               384,
    "BAAI/bge-m3":                                                    1024,
    "l3cube-pune/telugu-sentence-bert-nli":                           768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":    768,
    "intfloat/multilingual-e5-large":                                 1024,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":    384

}

# ── Search ────────────────────────────────────────────────────────────
TOP_K = 4   # candidates per vector space before fusion — higher than single-match
             # setups since we now want to surface EVERY plausible match, not just one
