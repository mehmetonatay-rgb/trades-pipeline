"""Stage 4 — load to Notion.

Idempotent upsert into two databases (Leads, Routes). The dedup key is `Place ID`:
Notion has no unique constraint, so the script enforces it by querying before insert.
On re-run, mutable fields are updated and `Status` is never overwritten (field-work
progress is preserved). Requests are throttled to ~3/sec with exponential backoff on
transient failures (429 rate limits, 5xx gateway/server errors, and network blips).
"""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx
from notion_client import Client
from notion_client.errors import APIResponseError, HTTPResponseError, RequestTimeoutError

# Transient HTTP statuses worth retrying — rate limit + gateway/server errors.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

from .cluster import single_place_maps_url
from .schemas import ClassifiedRecord, Cluster, PlaceRecord


# --------------------------------------------------------------------------- #
# Throttled client wrapper
# --------------------------------------------------------------------------- #
class NotionLoader:
    def __init__(self, config: dict, *, token: Optional[str] = None,
                 leads_db: Optional[str] = None, routes_db: Optional[str] = None):
        ncfg = config.get("notion", {})
        self.min_interval = 1.0 / float(ncfg.get("requests_per_sec", 3))
        self.max_retries = int(ncfg.get("max_retries", 5))
        self.client = Client(auth=token or os.environ["NOTION_TOKEN"])
        self.leads_db = leads_db or os.environ["NOTION_LEADS_DB_ID"]
        self.routes_db = routes_db or os.environ["NOTION_ROUTES_DB_ID"]
        self._last_call = 0.0

    def _call(self, fn, **kwargs):
        """Throttle + retry with exponential backoff on transient failures:
        HTTP 429 (rate limit), 5xx (gateway/server errors like the 502s Notion
        occasionally returns), and network timeouts/connection blips."""
        for attempt in range(self.max_retries):
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                result = fn(**kwargs)
                self._last_call = time.monotonic()
                return result
            except (APIResponseError, HTTPResponseError) as exc:
                status = getattr(exc, "status", None)
                self._last_call = time.monotonic()
                if status in _RETRYABLE_STATUS and attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * self.min_interval)
                    continue
                raise
            except (RequestTimeoutError, httpx.TransportError) as exc:
                # Network-level failure (timeout, connection reset) — retry.
                self._last_call = time.monotonic()
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * self.min_interval)
                    continue
                raise
        raise RuntimeError("Notion call exhausted retries")

    # ----- property builders -------------------------------------------------
    @staticmethod
    def _rich(text: str) -> list[dict]:
        return [{"text": {"content": text[:2000]}}] if text else []

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        if not phone:
            return ""
        digits = "".join(ch for ch in phone if ch.isdigit() or ch == "+")
        if digits.startswith("+"):
            return digits
        digits = digits.lstrip("0")
        if digits.startswith("90"):
            return "+" + digits
        return "+90" + digits

    def _lead_properties(self, cr: ClassifiedRecord, route_page_id: Optional[str]) -> dict:
        p = cr.place
        props: dict = {
            "Name": {"title": [{"text": {"content": p.name or "(isimsiz)"}}]},
            "Place ID": {"rich_text": self._rich(p.place_id)},
            "Trade": {"select": {"name": p.trade_label}} if p.trade_label else {"select": None},
            "Classification": {"select": {"name": cr.classification.label}},
            "Confidence": {"number": float(cr.classification.score)},
            "Address": {"rich_text": self._rich(p.address)},
            "Neighborhood": {"rich_text": self._rich(p.neighborhood)},
            "Source query": {"rich_text": self._rich(", ".join(p.source_queries))},
        }
        phone = self._normalize_phone(p.phone)
        if phone:
            props["Phone"] = {"phone_number": phone}
        if p.district:
            props["District"] = {"select": {"name": p.district}}
        if p.lat is not None:
            props["Latitude"] = {"number": p.lat}
        if p.lng is not None:
            props["Longitude"] = {"number": p.lng}
        maps_url = single_place_maps_url(p)
        if maps_url:
            props["Maps URL"] = {"url": maps_url}
        if p.rating is not None:
            props["Rating"] = {"number": p.rating}
        if p.review_count is not None:
            props["Reviews"] = {"number": p.review_count}
        if p.website:
            props["Website"] = {"url": p.website}
        if route_page_id:
            props["Route"] = {"relation": [{"id": route_page_id}]}
        return props

    # ----- lookups -----------------------------------------------------------
    def _find_lead(self, place_id: str) -> Optional[str]:
        resp = self._call(
            self.client.databases.query,
            database_id=self.leads_db,
            filter={"property": "Place ID", "rich_text": {"equals": place_id}},
            page_size=1,
        )
        results = resp.get("results", [])
        return results[0]["id"] if results else None

    def _find_route(self, route_id: str) -> Optional[str]:
        resp = self._call(
            self.client.databases.query,
            database_id=self.routes_db,
            filter={"property": "Name", "title": {"equals": route_id}},
            page_size=1,
        )
        results = resp.get("results", [])
        return results[0]["id"] if results else None

    # ----- upserts -----------------------------------------------------------
    def upsert_route(self, cluster: Cluster) -> tuple[str, bool]:
        props = {
            "Name": {"title": [{"text": {"content": cluster.route_id}}]},
            "District": {"select": {"name": cluster.district}},
            "Stops": {"number": len(cluster.ordered_place_ids)},
            "Distance km": {"number": cluster.distance_km},
        }
        if cluster.maps_urls:
            props["Maps link"] = {"url": cluster.maps_urls[0]}
        existing = self._find_route(cluster.route_id)
        if existing:
            self._call(self.client.pages.update, page_id=existing, properties=props)
            return existing, False
        props["Status"] = {"select": {"name": "Planned"}}
        page = self._call(self.client.pages.create,
                          parent={"database_id": self.routes_db}, properties=props)
        return page["id"], True

    def upsert_lead(self, cr: ClassifiedRecord, route_page_id: Optional[str]) -> bool:
        """Return True if a new page was created, False if an existing one was updated."""
        props = self._lead_properties(cr, route_page_id)
        existing = self._find_lead(cr.place.place_id)
        if existing:
            # Never overwrite Status — preserve field-work progress.
            self._call(self.client.pages.update, page_id=existing, properties=props)
            return False
        props["Status"] = {"select": {"name": "New"}}  # only on first insert
        self._call(self.client.pages.create,
                   parent={"database_id": self.leads_db}, properties=props)
        return True


def load_to_notion(
    kept: list[ClassifiedRecord],
    clusters: list[Cluster],
    config: dict,
) -> dict:
    """Upsert routes first (so leads can relate to them), then leads. Returns counts."""
    loader = NotionLoader(config)
    place_to_route: dict[str, str] = {}
    routes_new = 0

    for cluster in clusters:
        page_id, created = loader.upsert_route(cluster)
        routes_new += int(created)
        for pid in cluster.ordered_place_ids:
            place_to_route[pid] = page_id

    leads_new = leads_updated = 0
    for cr in kept:
        route_page_id = place_to_route.get(cr.place.place_id)
        created = loader.upsert_lead(cr, route_page_id)
        leads_new += int(created)
        leads_updated += int(not created)

    return {
        "routes_new": routes_new,
        "routes_total": len(clusters),
        "leads_new": leads_new,
        "leads_updated": leads_updated,
    }
