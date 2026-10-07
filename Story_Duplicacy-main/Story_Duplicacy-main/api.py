# api.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI app for the multi-client, multi-language news matcher.
# Run with:  uvicorn api:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import CLIENT_CONFIGS, RETENTION_DAYS
from matcher import NewsMatcher, UnknownClientError, StoryAlreadyExistsError, SOAPIngestError
from retention_scheduler import start_background_scheduler
from store import Story

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    client_name:  str  = Field(..., description=f"Must be one of: {list(CLIENT_CONFIGS.keys())}")
    story_id:     str  = Field(..., description="Your unique story ID — required, must not already exist")
    title:        str  = Field(..., description="Story headline in the publication's language")
    body:         str  = Field(..., description="Story body text")
    source:       Optional[str] = Field("", description="Byline / URL / agency tag")
    published_at: Optional[str] = Field("", description="ISO-8601 datetime; defaults to now if omitted")

    @field_validator("story_id")
    @classmethod
    def story_id_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("story_id must not be blank")
        return v.strip()

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title must not be blank")
        return v.strip()

    @field_validator("body")
    @classmethod
    def body_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("body must not be blank")
        return v.strip()


class MatchOut(BaseModel):
    story_id:           str
    title:              str
    body:               str
    source:             str
    published_at:        str
    fused_score:          float
    rerank_score:          float
    per_vector_scores:     dict


class IngestResponse(BaseModel):
    has_matches:    bool
    stored_id:      str         # the caller's own story_id, echoed back
    match_count:    int
    matches:        list[MatchOut]
    message:        str


# ── App lifecycle ─────────────────────────────────────────────────────

_matcher: Optional[NewsMatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _matcher
    logger.info("Starting up — loading all client models (this may take a while)...")
    _matcher = NewsMatcher()
    start_background_scheduler(_matcher, interval_hours=24)
    logger.info("Startup complete. Clients: %s", list(CLIENT_CONFIGS.keys()))
    yield


app = FastAPI(
    title       = "Multi-Publication News Event Matcher",
    description = "Detects same-event stories within each publication's own history. Returns ALL matches above threshold.",
    version     = "3.0.0",
    lifespan    = lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model = IngestResponse,
    summary        = "Ingest a story; returns ALL same-event matches",
    responses      = {
        400: {"description": "Blank story_id, unknown client_name, or story_id already exists"},
        502: {"description": "SOAP save-pair call failed — story was NOT stored"},
    },
)
def ingest_story(req: IngestRequest):
    """
    Submit a story for a specific client.

    **Error cases:**
    -HTTP 400
     - `story_id` is blank or whitespace-only
     - `client_name` is not in the configured client list
     - `story_id` already exists in the database

    - `502` — SOAP API call failed for one or more matched pairs;
      the story was **not** stored (safe to retry after fixing the SOAP issue)

    **On success:**
    - Returns every previously stored story that clears this client's
      similarity + rerank thresholds (can be zero, one, or many)
    - The story is always stored, whether or not matches were found
    - All matched pairs were successfully sent to the SOAP API before storage
    """
    if req.client_name not in CLIENT_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown client_name '{req.client_name}'. Known clients: {list(CLIENT_CONFIGS.keys())}",
        )

    story = Story(
        story_id     = req.story_id,
        client_name  = req.client_name,
        title        = req.title,
        body         = req.body,
        source       = req.source or "",
        published_at = req.published_at or "",
    )

    try:
        result = _matcher.ingest(story)
    except UnknownClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StoryAlreadyExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SOAPIngestError as e:
        # Story was NOT stored — safe for the caller to retry once SOAP is fixed
        raise HTTPException(status_code=502, detail=str(e))

    matches_out = [
        MatchOut(
            story_id          = r.candidate.story.story_id,
            title             = r.candidate.story.title,
            body              = r.candidate.story.body,
            source            = r.candidate.story.source,
            published_at      = r.candidate.story.published_at,
            fused_score       = round(r.candidate.fused_score, 4),
            rerank_score      = round(r.rerank_score, 4),
            per_vector_scores = {k: round(v, 4) for k, v in r.candidate.per_vector_scores.items()},
        )
        for r in result.matches
    ]

    message = (
        f"Found {len(matches_out)} same-event match(es)."
        if result.has_matches else
        "No matching event found. Story stored."
    )

    return IngestResponse(
        has_matches  = result.has_matches,
        stored_id    = result.stored_id,
        match_count  = len(matches_out),
        matches      = matches_out,
        message      = message,
    )


@app.get("/stats", summary="Story counts, overall and per-client")
def stats():
    return {
        "total_stories": _matcher.count(),
        "per_client": {name: _matcher.count(name) for name in CLIENT_CONFIGS.keys()},
    }


@app.post("/admin/retention-sweep", summary="Manually trigger the retention cleanup")
def retention_sweep():
    """
    Hard-deletes stories older than RETENTION_DAYS, across ALL clients.
    In production, call this on a schedule (cron job, or a background
    task scheduler) rather than manually.
    """
    deleted = _matcher.run_retention_sweep(RETENTION_DAYS)
    return {"deleted": deleted, "retention_days": RETENTION_DAYS}


@app.get("/clients", summary="List configured clients and their settings")
def list_clients():
    return {
        name: {
            "language": cfg["language"],
            "models": [m[1] for m in cfg["models"]],
            "similarity_threshold": cfg["similarity_threshold"],
            "rerank_threshold": cfg["rerank_threshold"],
        }
        for name, cfg in CLIENT_CONFIGS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok"}
