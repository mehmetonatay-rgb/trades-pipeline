"""Stage 5 — orchestrate + report + CLI.

Runs the sequential pipeline: config → fetch → classify → cluster → load(Notion) → report.
Each stage writes its normalized output to data/ so any stage can be re-run / debugged
in isolation.

    python -m src.pipeline --district Kadıköy --trades electrician,plumber
    python -m src.pipeline --refresh            # ignore cache, re-hit the API
    python -m src.pipeline --no-notion          # dry run to data/ only
    python -m src.pipeline --llm-uncertain      # LLM pass on the Uncertain band
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import classify as classify_mod
from . import cluster as cluster_mod
from . import fetch as fetch_mod
from .schemas import ClassifiedRecord, Cluster, PlaceRecord

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _select_trades(config: dict, trade_keys: list[str] | None) -> list[dict]:
    trades = config["trades"]
    if not trade_keys:
        return trades
    wanted = set(trade_keys)
    selected = [t for t in trades if t["key"] in wanted]
    missing = wanted - {t["key"] for t in selected}
    if missing:
        raise SystemExit(f"Unknown trade key(s): {', '.join(sorted(missing))}")
    return selected


def _dump_json(name: str, payload) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_fetch(config, districts, trades, refresh) -> list[PlaceRecord]:
    records = fetch_mod.run_fetch(config, districts, trades, refresh=refresh)
    _dump_json("01_fetched.json", [r.to_dict() for r in records])
    return records


def stage_classify(config, records, use_llm) -> list[ClassifiedRecord]:
    classified = classify_mod.classify_records(records, config, use_llm=use_llm)
    _dump_json("02_classified.json", [c.to_dict() for c in classified])
    return classified


def stage_cluster(config, kept) -> list[Cluster]:
    clusters = cluster_mod.cluster_leads(kept, config)
    _dump_json("03_clusters.json", [c.to_dict() for c in clusters])
    return clusters


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(
    fetched: list[PlaceRecord],
    classified: list[ClassifiedRecord],
    clusters: list[Cluster],
    notion_counts: dict | None,
) -> str:
    per_trade: dict[str, Counter] = defaultdict(Counter)
    for c in classified:
        per_trade[c.place.trade_label or c.place.trade_key][c.classification.label] += 1

    lines = ["", "=" * 60, "TRADES LEAD PIPELINE — REPORT", "=" * 60]
    lines.append(f"Fetched (deduped): {len(fetched)}")
    lines.append("")
    lines.append(f"{'Trade':<22}{'Service':>9}{'Supply':>9}{'Uncert.':>9}")
    lines.append("-" * 49)
    totals = Counter()
    for trade in sorted(per_trade):
        c = per_trade[trade]
        totals.update(c)
        lines.append(
            f"{trade:<22}{c['Service']:>9}{c['Supply']:>9}{c['Uncertain']:>9}"
        )
    lines.append("-" * 49)
    lines.append(
        f"{'TOTAL':<22}{totals['Service']:>9}{totals['Supply']:>9}{totals['Uncertain']:>9}"
    )

    stops = [len(cl.ordered_place_ids) for cl in clusters]
    avg_stops = (sum(stops) / len(stops)) if stops else 0
    lines.append("")
    lines.append(f"Routes: {len(clusters)}  |  avg stops/route: {avg_stops:.1f}")

    if notion_counts is not None:
        lines.append("")
        lines.append(
            f"Notion leads — new: {notion_counts['leads_new']}, "
            f"updated: {notion_counts['leads_updated']}"
        )
        lines.append(
            f"Notion routes — new: {notion_counts['routes_new']}, "
            f"total: {notion_counts['routes_total']}"
        )
    else:
        lines.append("\nNotion: skipped (--no-notion). Output in data/.")
    lines.append("=" * 60)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Trades Lead Pipeline")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--district", help="Comma-separated districts (default: all in config)")
    parser.add_argument("--trades", help="Comma-separated trade keys (default: all in config)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, re-hit the API")
    parser.add_argument("--no-notion", action="store_true", help="Dry run to data/ only")
    parser.add_argument("--llm-uncertain", action="store_true", help="LLM pass on Uncertain band")
    parser.add_argument("--tile", action="store_true",
                        help="Grid-tile each district's bbox to beat the 60-result cap (Google only)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config(Path(args.config))
    if args.tile:
        config.setdefault("fetch", {}).setdefault("tiling", {})["enabled"] = True

    districts = (
        [d.strip() for d in args.district.split(",")] if args.district else config["districts"]
    )
    trade_keys = [t.strip() for t in args.trades.split(",")] if args.trades else None
    trades = _select_trades(config, trade_keys)

    print(f"→ Fetch: {len(districts)} district(s) × {len(trades)} trade(s) "
          f"[source={config.get('source')}]")
    fetched = stage_fetch(config, districts, trades, args.refresh)

    print(f"→ Classify: {len(fetched)} record(s)"
          + (" (+LLM on Uncertain)" if args.llm_uncertain else ""))
    classified = stage_classify(config, fetched, args.llm_uncertain)

    kept = [c for c in classified if c.classification.label == "Service"]
    print(f"→ Cluster: {len(kept)} kept lead(s)")
    clusters = stage_cluster(config, kept)

    notion_counts = None
    if not args.no_notion:
        print(f"→ Load to Notion: {len(kept)} lead(s), {len(clusters)} route(s)")
        from . import notion_load
        notion_counts = notion_load.load_to_notion(kept, clusters, config)

    print(build_report(fetched, classified, clusters, notion_counts))


if __name__ == "__main__":
    main()
