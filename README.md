# Trades Lead Pipeline (v1)

A no-UI pipeline that finds tradespeople who **perform** on-site work (electricians,
plumbers, …) in Istanbul districts, filters out shops that merely **sell materials**,
clusters the survivors into tight door-to-door routes, and upserts everything into Notion.

```
config → fetch → classify → cluster → load(Notion) → report
            │         │          │
          cache/   data/      data/
```

Each stage reads the previous stage's output from disk, so any stage can be re-run alone.
`cache/` holds raw API responses (tuning the classifier never re-hits the paid API);
`data/` holds normalized intermediate JSON for debugging.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys
```

Edit `config.yaml` — districts, trades, query terms, and all classifier/cluster weights
live there. No logic is hardcoded.

### Create the Notion databases (once)

Share a parent page with your Notion integration, then:

```bash
python -m src.notion_setup --parent-page <PAGE_ID>
```

Copy the printed `NOTION_LEADS_DB_ID` / `NOTION_ROUTES_DB_ID` into `.env`.

## Run

```bash
# all districts/trades from config, write to Notion
python -m src.pipeline

# scope it
python -m src.pipeline --district Kadıköy --trades electrician,plumber

# dry run — no Notion, just data/ + the report
python -m src.pipeline --no-notion

# force-refresh the cache (re-hit the API)
python -m src.pipeline --refresh

# add the cost-controlled LLM pass on the Uncertain band
python -m src.pipeline --llm-uncertain
```

## How classification works (the heart of v1)

A score combines three signals (see `src/classify.py` and the `classify:` block in
`config.yaml`):

| Signal | Effect |
|---|---|
| Service-intent query terms | applied at **fetch** time as seeds (`elektrikçi`, never `elektrik malzemeleri`) |
| Category allow/block | `+2` service category, `-3` supply/retail category |
| Name keywords | `+1` per service word (cap `+2`), `-2` per supply word (cap `-4`) |
| Website | `+1` if no website or not an e-commerce/catalog domain |

`score >= 2` → **Service** (keep) · `score <= -2` → **Supply** (dropped but logged) ·
otherwise **Uncertain**. Tune the weights in `config.yaml` until Service precision
is ≥85% on a hand-checked sample of ~25 keeps per trade.

## Idempotency

The Notion loader dedups on `Place ID` (Notion has no unique constraint, so the script
enforces it): re-running updates mutable fields and **never overwrites `Status`**, so
field-work progress is preserved. Routes dedup on their `route_id` title.

## Source adapters

`source: google` (default) uses the Google Places API (New) `places:searchText` with a
field mask to control cost. `source: apify` uses the Apify Google Maps scraper — same
`PlaceRecord` shape downstream. Switch in `config.yaml`.

## Layout

```
config.yaml          # everything tunable
.env                 # secrets (copy from .env.example)
cache/               # raw API responses, keyed by {source}_{district}_{term}
data/                # normalized intermediate JSON between stages
src/
  schemas.py         # PlaceRecord + classified/cluster records
  fetch.py           # google + apify adapters, caching, dedup
  classify.py        # scoring + optional LLM
  cluster.py         # DBSCAN + nearest-neighbour routing + maps URLs
  notion_load.py     # idempotent upsert, dedup, throttling
  notion_setup.py    # one-time DB creation
  pipeline.py        # orchestrates + report + CLI
```

See `trades_lead_pipeline_spec.md` for the full spec and extension points.
