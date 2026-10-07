# matcher.py
# ─────────────────────────────────────────────────────────────────────
# Orchestrates: validate → duplicate-check → embed → search → fuse →
# rerank → SOAP save pairs → store to Qdrant.
#
# SOAP is called BEFORE store() so that on failure there is nothing
# to roll back — the story simply never enters Qdrant.
#
# Sequence on a story with N matches:
#   1. Validate client_name
#   2. Check story_id not already in Qdrant  (O(1) lookup, fail fast)
#   3. Embed using this client's models
#   4. Search + fuse across vector spaces
#   5. Rerank candidates
#   6. Call SOAP save_story_pair() for EVERY match
#        → if ANY call fails: raise SOAPIngestError (nothing stored)
#   7. Store story to Qdrant  (only reached if step 6 fully succeeded)
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass, field

from config import CLIENT_CONFIGS
from embedder import Embedder
from reranker import Reranker, RankedCandidate
from soap_client import save_story_pair, SOAPCallError
from store import Story, StoryStore

logger = logging.getLogger(__name__)


class UnknownClientError(ValueError):
    pass

class StoryAlreadyExistsError(ValueError):
    pass

class SOAPIngestError(RuntimeError):
    """
    Raised when one or more SOAP save_story_pair calls fail.
    The story has NOT been stored in Qdrant when this is raised.
    """
    pass

@dataclass
class IngestResult:
    incoming:    Story
    matches:     list           # list[RankedCandidate]
    stored_id:   str            # caller's story_id (not the Qdrant UUID)
    has_matches: bool = field(init=False)


    def __post_init__(self):
        self.has_matches = len(self.matches) > 0


class NewsMatcher:

    def __init__(self):
        self.embedder = Embedder()          # loads every model used by any client, once each
        self.store    = StoryStore(self.embedder)
        self.reranker = Reranker()           # loads every reranker used by any client, once each
        logger.info("NewsMatcher ready for clients: %s", list(CLIENT_CONFIGS.keys()))

    def ingest(self, story: Story) -> IngestResult:
        # ── 1. Validate client ─────────────────────────────────────────
        if story.client_name not in CLIENT_CONFIGS:
            raise UnknownClientError(
                f"Unknown client_name '{story.client_name}'. "
                f"Known clients: {list(CLIENT_CONFIGS.keys())}"
            )

        # ── 2. Duplicate check — BEFORE embedding (cheap O(1) lookup) ─
        if self.store.exists(story.story_id):
            raise StoryAlreadyExistsError(
                f"story_id '{story.story_id}' already exists in the database."
            )

        text = f"{story.title}. {story.body}"

        # ── 3. Embed using this client's configured models ─────────────
        vectors = self.embedder.embed_for_client(story.client_name, text)

        # ── 4. Search + fuse across this client's vector spaces ────────
        candidates = self.store.search(story, vectors)
        logger.info("[%s] %d candidates after search + fusion", story.client_name, len(candidates))

        # ── 5. Rerank — keep ALL above client's rerank_threshold ───────
        ranked: list[RankedCandidate] = self.reranker.rerank(story.client_name, text, candidates)

        # 6. SOAP — call for every match, BEFORE storing to Qdrant
        if ranked:
            self._soap_save_all(story.story_id, ranked)
            # raises SOAPIngestError on any failure — story not stored yet

        # 7. Store to Qdrant (only reached if all SOAP calls succeeded)
        story.matched_story_ids = [r.candidate.story.story_id for r in ranked]
        self.store.store(story, vectors)

        return IngestResult(
            incoming  = story,
            matches   = ranked,
            stored_id = story.story_id,     # return CALLER's ID, not Qdrant UUID
        )

    def _soap_save_all(self, incoming_id: str, ranked: list[RankedCandidate]) -> None:
        """
        Call save_story_pair(incoming_id, matched_id) for every ranked match.
        Fail-fast: raises SOAPIngestError on the first failure.

        To attempt ALL pairs and report all failures instead, uncomment
        the "collect all errors" variant below and remove the loop above it.
        """
        for rc in ranked:
            matched_id = rc.candidate.story.story_id
            try:
                save_story_pair(incoming_id, matched_id)
            except SOAPCallError as e:
                raise SOAPIngestError(
                    f"SOAP call failed for pair ('{incoming_id}', '{matched_id}'). "
                    f"Story was NOT stored. Original error: {e}"
                ) from e

        # ── "Collect all errors" variant ────────────────────────────────
        # errors = []
        # for rc in ranked:
        #     matched_id = rc.candidate.story.story_id
        #     try:
        #         save_story_pair(incoming_id, matched_id)
        #     except SOAPCallError as e:
        #         errors.append(str(e))
        # if errors:
        #     raise SOAPIngestError(
        #         f"SOAP failed for {len(errors)} pair(s). Story NOT stored.\n" +
        #         "\n".join(errors)
        #     )

    def count(self, client_name: str = None) -> int:
        return self.store.count(client_name)

    def run_retention_sweep(self, days: int) -> int:
        return self.store.delete_older_than(days)
