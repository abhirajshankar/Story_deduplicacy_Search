# embedder.py
# ─────────────────────────────────────────────────────────────────────
# Loads embedding models ONCE across all clients (deduplicated — if two
# clients share a model like bge-m3, it's loaded into memory only once),
# and exposes per-client embedding via the client's configured model set.
# ─────────────────────────────────────────────────────────────────────

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer

from config import CLIENT_CONFIGS, MODEL_DIMS

logger = logging.getLogger(__name__)

# Models that require e5-style "query: " prefixing
E5_PREFIX_MODELS = {"intfloat/multilingual-e5-large", "intfloat/multilingual-e5-base"}


class Embedder:

    def __init__(self):
        self._models: dict[str, SentenceTransformer] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        """Load every distinct model referenced across all client configs, once each."""
        unique_models: set[str] = set()
        for client_cfg in CLIENT_CONFIGS.values():
            for _, model_name in client_cfg["models"]:
                unique_models.add(model_name)

        logger.info("Loading %d distinct embedding models (shared across clients)...", len(unique_models))
        for model_name in unique_models:
            logger.info("  loading %s", model_name)
            self._models[model_name] = SentenceTransformer(model_name)
        logger.info("All embedding models loaded.")

    def embed_for_client(self, client_name: str, text: str) -> dict[str, list[float]]:
        """
        Embed `text` using only the models configured for `client_name`.
        Returns { vector_name: embedding } using that client's vector-name scheme.
        """
        client_cfg = CLIENT_CONFIGS[client_name]
        results: dict[str, list[float]] = {}

        def _run(vec_name: str, model_name: str) -> tuple[str, list[float]]:
            input_text = f"query: {text}" if model_name in E5_PREFIX_MODELS else text
            model = self._models[model_name]
            vector = model.encode(input_text, normalize_embeddings=True).tolist()
            return vec_name, vector

        models = client_cfg["models"]
        with ThreadPoolExecutor(max_workers=max(len(models), 1)) as pool:
            futures = {
                pool.submit(_run, vec_name, model_name): vec_name
                for vec_name, model_name in models
            }
            for future in as_completed(futures):
                vec_name, vector = future.result()
                results[vec_name] = vector

        return results

    @staticmethod
    def vector_dims_for_client(client_name: str) -> dict[str, int]:
        """Returns { vec_name: dimension } for this client's configured vectors."""
        client_cfg = CLIENT_CONFIGS[client_name]
        return {
            vec_name: MODEL_DIMS[model_name]
            for vec_name, model_name in client_cfg["models"]
        }

    @staticmethod
    def all_vector_dims() -> dict[str, int]:
        """Returns { vec_name: dimension } across ALL clients — used to build the
        shared Qdrant collection schema, which must declare every named vector
        up front even though any single point only populates its own client's subset."""
        dims: dict[str, int] = {}
        for client_cfg in CLIENT_CONFIGS.values():
            for vec_name, model_name in client_cfg["models"]:
                dims[vec_name] = MODEL_DIMS[model_name]
        return dims
