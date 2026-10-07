# Multi-Publication News Event Matcher

Single shared Qdrant collection serving multiple publications (PTI, Eenadu,
Dainik Jagran, Lokmat, ...), each in its own language, with matching strictly
scoped within each publication. Returns ALL same-event matches above
threshold, not just the closest one.

## Architecture summary

```
ONE Qdrant collection: "news_stories"
   │
   ├── client_name = "PTI"            (English, hard filter)
   ├── client_name = "Eenadu"         (Telugu,  hard filter)
   ├── client_name = "Dainik Jagran"  (Hindi,   hard filter)
   └── client_name = "Lokmat"         (Marathi, hard filter)

Each story stores ONLY the named vectors relevant to its client's
configured models (Qdrant allows points to have a subset of a
collection's declared named vectors).

Matching NEVER crosses client_name — every search has
  must: client_name == <this story's client>
  must: published_ts within ±3 days
as hard filters before any vector search happens.
```

## Why one collection, not one per client

Qdrant collections carry per-collection index overhead. With ~5+
publications you'd be running 5+ partially-empty HNSW indexes instead of
one shared one. At your volume (~1500 stories/day, 3-day retention ≈ 4500
points total) this barely matters for performance — the real win is
operational simplicity: one collection schema, one backup, one place to
look. If a future requirement needs different *vector schemas* (not just
different filters) per client, that's the point to reconsider — but
filtering covers everything you described.

## Setup

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

First startup loads every distinct model referenced across all clients —
shared models (e.g. `BAAI/bge-m3` used by Eenadu, Dainik Jagran, and
Lokmat) are loaded ONCE, not once per client. Expect a slow first run
(multiple GB of model weights) and a few GB of RAM at runtime.

## Adding or editing a client

Everything lives in `config.py` → `CLIENT_CONFIGS`. To add a new
publication:

```python
"Amar Ujala": {
    "language": "hi",
    "models": [
        ("vec_amarujala_bge_m3", "BAAI/bge-m3"),
    ],
    "weights": {"vec_amarujala_bge_m3": 1.0},
    "similarity_threshold": 0.72,   # placeholder — calibrate with real data
    "rerank_threshold": 0.30,       # placeholder — calibrate with real data
    "reranker_model": DEFAULT_RERANKER_MODEL,
},
```

Vector names must be globally unique across the whole config (they live in
one shared Qdrant collection schema) — prefixing with the client name as
shown avoids collisions.

## Calibrating thresholds per client (important — do this before production)

Cosine similarity scores from the same model are NOT comparable across
languages. A 0.80 threshold that works for PTI's English embeddings will
behave completely differently for Eenadu's Telugu embeddings. For each
client:

1. Collect 20-30 labeled pairs: known duplicates (same event, different
   wording) and known non-duplicates (different events, similar topic)
2. Run them through `/ingest` and look at `fused_score` and
   `rerank_score` in the response
3. Find the score that best separates true matches from false ones
4. Update that client's `similarity_threshold` / `rerank_threshold` in
   `config.py`

The defaults currently in `config.py` are reasonable starting guesses,
not measured values.

## API

### POST /ingest
```json
{
  "client_name": "Eenadu",
  "title": "...",
  "body": "...",
  "source": "Eenadu",
  "published_at": "2024-06-01T10:00:00Z"
}
```

Returns ALL matches above threshold:
```json
{
  "has_matches": true,
  "stored_id": "uuid-of-new-story",
  "match_count": 3,
  "matches": [
    {"story_id": "...", "title": "...", "fused_score": 0.91, "rerank_score": 0.84, ...},
    {"story_id": "...", "title": "...", "fused_score": 0.88, "rerank_score": 0.79, ...},
    {"story_id": "...", "title": "...", "fused_score": 0.85, "rerank_score": 0.71, ...}
  ],
  "message": "Found 3 same-event match(es)."
}
```

Unknown `client_name` → `400 Bad Request` with the list of valid clients
(prevents typos from silently creating an orphaned matching universe).

The incoming story is **always stored**, whether or not matches were
found. Its `matched_story_ids` payload field records what it matched —
useful later for tracing all variants of one real-world event.

### GET /stats
Total stories, plus a per-client breakdown.

### GET /clients
Lists every configured client with its language, models, and thresholds —
useful for sanity-checking config without opening the file.

### POST /admin/retention-sweep
Manually triggers a hard delete of stories older than `RETENTION_DAYS`
(default 3). Also runs automatically every 24 hours in-process (see
`retention_scheduler.py`) — manual trigger is for testing or one-off cleanup.

## Retention

`RETENTION_DAYS = 3` in `config.py`. Two ways it runs:

- **Automatic (default):** `start_background_scheduler()` runs inside the
  FastAPI process via a daemon thread, sweeping every 24 hours.
- **Cron-based (more robust):** run `python retention_scheduler.py --once`
  on a schedule instead. ⚠️ Do not run this standalone version while the
  API is also running against the same `QDRANT_PATH` — Qdrant's local
  (`path=`) mode only allows one process to hold the lock at a time. If
  you want cleanup decoupled from the API process, switch Qdrant to
  server mode (`QdrantClient(url=...)` against a running Qdrant server)
  instead of local path mode.

## What's deliberately NOT handled here (flagging so it's a choice, not an accident)

- **Cross-publication matching** — explicitly out of scope per your
  requirements. If this changes later, it's a bigger redesign (shared
  multilingual embedding space, no more hard client_name filter).
- **Queue-based async ingestion** — at ~1500 stories/day combined,
  synchronous request/response per story is fine. If volume grows
  10x+, revisit with a task queue (Celery/RQ) in front of `/ingest`.
- **Translation** — not used anywhere in this version; every client
  embeds directly in its own language.
