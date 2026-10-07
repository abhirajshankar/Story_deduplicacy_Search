# store.py
# ─────────────────────────────────────────────────────────────────────
# Qdrant wrapper for the multi-client matcher.
#
# ONE shared collection. Every search/store call is scoped to a single
# client_name via a hard filter — clients never see each other's stories.
# Returns ALL candidates above threshold (not just the best one).

# IMPORTANT — Qdrant point ID constraints:
# Qdrant only accepts unsigned integers or valid UUIDs as point IDs.
# Arbitrary strings like "PTI-2024-001" are rejected with a UUID error.
#
# Solution: the CALLER's story_id is stored in the payload (as a plain
# string — no restrictions there), and we derive the actual Qdrant point
# ID using uuid.uuid5(), which is DETERMINISTIC:
#
#   qdrant_id = uuid.uuid5(NAMESPACE, caller_story_id)
#
# Same caller story_id → always same Qdrant UUID → we can check for
# existence (duplicate detection) cheaply with a single retrieve() call
# before doing any embedding work.
# ─────────────────────────────────────────────────────────────────────

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, NamedVector, PointStruct,
    Filter, FieldCondition, Range, MatchValue,
)

from config import QDRANT_PATH, COLLECTION_NAME, TOP_K, TIME_WINDOW_DAYS, CLIENT_CONFIGS
from embedder import Embedder

logger = logging.getLogger(__name__)



# Fixed namespace for uuid5 derivation — do NOT change after you have data,
# or all existing Qdrant IDs will mismatch.
_UUID5_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def caller_id_to_qdrant_id(story_id: str) -> str:
    """
    Derive a deterministic Qdrant-compatible UUID from an arbitrary caller story_id.
    The same story_id always produces the same UUID, making duplicate detection trivial.
    """
    return str(uuid.uuid5(_UUID5_NAMESPACE, story_id))



@dataclass
class Story:
    title:        str
    body:         str
    client_name:  str                  # e.g. "PTI", "Eenadu" — must be a known client
    story_id:     str           # REQUIRED — caller's own ID (any string)
    source:       str = ""
    published_at: str = ""             # ISO-8601
    matched_story_ids: list = field(default_factory=list)   # filled in after matching


@dataclass
class Candidate:
    story:             Story
    fused_score:        float
    per_vector_scores:  dict = field(default_factory=dict)


class StoryStore:

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.client   = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()
        logger.info("StoryStore ready — collection '%s' has %d total stories.",
                     COLLECTION_NAME, self.count())

    # ── Public ────────────────────────────────

    def exists(self, story_id: str) -> bool:
        """
        Check if a story with this caller story_id already exists.
        Uses the derived Qdrant UUID for an O(1) point lookup — no scan needed.
        """
        qdrant_id = caller_id_to_qdrant_id(story_id)
        results = self.client.retrieve(
            collection_name = COLLECTION_NAME,
            ids             = [qdrant_id],
            with_payload    = False,   # we only care about existence, not content
        )
        return len(results) > 0

    def search(self, story: Story, vectors: dict[str, list[float]]) -> list[Candidate]:
        """
        Search ONLY within story.client_name's own stories, within the time window.
        Returns ALL candidates whose fused score clears that client's similarity_threshold —
        not just the top one.
        """
        client_cfg = CLIENT_CONFIGS[story.client_name]
        threshold  = client_cfg["similarity_threshold"]

        base_filter = self._build_filter(story.client_name, story.published_at)

        per_space: dict[str, dict[str, float]] = {}
        for vec_name, vec in vectors.items():
            hits = self.client.query_points(
                collection_name = COLLECTION_NAME,
                query=vec,
                using=vec_name,
                query_filter    = base_filter,
                limit           = TOP_K,
                with_payload    = True,
                score_threshold = threshold,
            )
            # per_space[vec_name] = {h.payload["story_id"]: h.score for h in hits}
            per_space[vec_name] = {
                                    p.payload["story_id"]: p.score
                                    for p in hits.points
                                }

        return self._fuse(story.client_name, per_space)

    def store(self, story: Story, vectors: dict[str, list[float]]) -> None:
        """
        Persist the story regardless of whether it matched anything.
        matched_story_ids is set by the caller (matcher.py) BEFORE calling store(),
        so the link is recorded even though we always store.

        Qdrant point ID is derived from story.story_id (the caller's ID).
        The caller's original story_id is also stored in the payload for retrieval.
        """
        qdrant_id = caller_id_to_qdrant_id(story.story_id)
        ts = self._iso_to_ts(story.published_at)
        self.client.upsert(
            collection_name = COLLECTION_NAME,
            points = [
                PointStruct(
                    id      = qdrant_id,        # Qdrant UUID (derived, always valid)
                    vector  = vectors,
                    payload = {
                        "story_id"           : story.story_id,      # CALLER's original ID
                        "client_name"        : story.client_name,
                        "title"              : story.title,
                        "body"               : story.body,
                        "source"             : story.source,
                        "published_at"       : story.published_at,
                        "published_ts"       : ts,
                        "ingested_at"        : int(time.time()),
                        "matched_story_ids"  : story.matched_story_ids,
                    },
                )
            ],
        )
        logger.info("Stored [%s] story_id='%s'  qdrant_id=%s  matched=%d",
            story.client_name, story.story_id, qdrant_id, len(story.matched_story_ids)
        )

    def count(self, client_name: Optional[str] = None) -> int:
        if client_name is None:
            return self.client.get_collection(COLLECTION_NAME).points_count or 0
        # count with filter requires a scroll/count call
        result = self.client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(must=[FieldCondition(key="client_name", match=MatchValue(value=client_name))]),
        )
        return result.count

    def delete_older_than(self, days: int) -> int:
        """Hard-delete every point across ALL clients older than `days`. Returns count deleted (approx)."""
        cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        before = self.count()
        self.client.delete(
            collection_name = COLLECTION_NAME,
            points_selector  = Filter(
                must=[FieldCondition(key="published_ts", range=Range(lt=cutoff_ts))]
            ),
        )
        after = self.count()
        deleted = before - after
        logger.info("Retention sweep: deleted %d points older than %d days.", deleted, days)
        return deleted

    # ── Internals ─────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            return

        vectors_config = {
            vec_name: VectorParams(size=dim, distance=Distance.COSINE)
            for vec_name, dim in self.embedder.all_vector_dims().items()
        }
        self.client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)

        # Index payload fields used in filters — required for efficient filtering at scale
        self.client.create_payload_index(COLLECTION_NAME, field_name="client_name", field_schema="keyword")
        self.client.create_payload_index(COLLECTION_NAME, field_name="published_ts", field_schema="integer")

        logger.info("Created Qdrant collection '%s' with %d named vectors.",
                     COLLECTION_NAME, len(vectors_config))

    def _fuse(self, client_name: str, per_space: dict[str, dict[str, float]]) -> list[Candidate]:
        weights = CLIENT_CONFIGS[client_name]["weights"]

        all_caller_ids: set[str] = set()
        for scores in per_space.values():
            all_caller_ids.update(scores.keys())
        if not all_caller_ids:
            return []

        # Retrieve full payloads using derived Qdrant UUIDs
        qdrant_ids = [caller_id_to_qdrant_id(sid) for sid in all_caller_ids]
        results    = self.client.retrieve(collection_name=COLLECTION_NAME, ids=qdrant_ids, with_payload=True)
        # Re-key by caller story_id (from payload) for score join
        payload_by_caller_id = {r.payload["story_id"]: r.payload for r in results}

        candidates: list[Candidate] = []
        for caller_id in all_caller_ids:
            weighted_sum, total_weight, per_vec_scores = 0.0, 0.0, {}
            for vec_name, scores in per_space.items():
                if caller_id in scores:
                    w = weights.get(vec_name, 1.0)
                    weighted_sum += scores[caller_id] * w
                    total_weight += w
                    per_vec_scores[vec_name] = scores[caller_id]

            fused_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            p = payload_by_caller_id.get(caller_id, {})
            story = Story(
                story_id          = p.get("story_id", caller_id),
                client_name       = p.get("client_name", client_name),
                title             = p.get("title", ""),
                body              = p.get("body", ""),
                source            = p.get("source", ""),
                published_at      = p.get("published_at", ""),
                matched_story_ids = p.get("matched_story_ids", []),
            )
            candidates.append(Candidate(
                story             = story,
                fused_score       = fused_score,
                per_vector_scores = per_vec_scores,
            ))

        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates

    def _build_filter(self, client_name: str, published_at: str) -> Filter:
        must = [FieldCondition(key="client_name", match=MatchValue(value=client_name))]

        if TIME_WINDOW_DAYS and published_at:
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                lo = int((dt - timedelta(days=TIME_WINDOW_DAYS)).timestamp())
                hi = int((dt + timedelta(days=TIME_WINDOW_DAYS)).timestamp())
                must.append(FieldCondition(key="published_ts", range=Range(gte=lo, lte=hi)))
            except Exception as e:
                logger.warning("Could not build time filter for '%s': %s", published_at, e)

        return Filter(must=must)

    @staticmethod
    def _iso_to_ts(published_at: str) -> int:
        if not published_at:
            return int(time.time())   # default to "now" if not provided
        try:
            return int(datetime.fromisoformat(published_at.replace("Z", "+00:00")).timestamp())
        except Exception:
            return int(time.time())
