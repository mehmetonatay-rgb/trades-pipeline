"""Stage 1 — fetch.

Pull businesses per (district × query_term) and normalize into PlaceRecord.
Two interchangeable adapters behind one interface:

    fetch(query, district) -> list[PlaceRecord]

Raw API responses are cached under cache/{source}_{district}_{term}.json so tuning
the classifier or clusterer never re-hits the paid API. Pass refresh=True to force.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import httpx

from .schemas import PlaceRecord

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

GOOGLE_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Field mask — request only what downstream needs, to control cost.
GOOGLE_FIELD_MASK = ",".join(
    "places." + f
    for f in [
        "id",
        "displayName",
        "primaryType",
        "primaryTypeDisplayName",
        "types",
        "formattedAddress",
        "location",
        "rating",
        "userRatingCount",
        "websiteUri",
        "nationalPhoneNumber",
        "businessStatus",
        "addressComponents",
        # NOTE: `reviews` is intentionally omitted — it's part of Google's pricier
        # Enterprise+Atmosphere SKU and is gated separately (causes intermittent 403s).
        # The optional LLM pass tolerates empty review_snippets. If you need snippets,
        # fetch them via Place Details for the Uncertain band only.
    ]
) + ",nextPageToken"


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    return re.sub(r"[^0-9a-zçğıöşü_-]", "", value)


def _cache_path(source: str, district: str, term: str) -> Path:
    return CACHE_DIR / f"{source}_{_slug(district)}_{_slug(term)}.json"


def _read_cache(path: Path) -> Optional[list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _write_cache(path: Path, raw_items: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw_items, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Address component parsing (Turkey: il > ilçe > mahalle)
# --------------------------------------------------------------------------- #
_DISTRICT_TYPES = ("administrative_area_level_2",)
_NEIGHBORHOOD_TYPES = ("neighborhood", "sublocality_level_1", "administrative_area_level_4")


def _pick_component(components: Iterable[dict], wanted_types: tuple[str, ...]) -> str:
    for comp in components or []:
        comp_types = comp.get("types", [])
        if any(t in comp_types for t in wanted_types):
            return comp.get("longText") or comp.get("long_name") or ""
    return ""


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #
class FetchAdapter(ABC):
    source: str = "base"

    @abstractmethod
    def fetch_raw(self, query: str, district: str) -> list[dict]:
        """Return the raw, source-shaped place dicts for one (query, district)."""

    @abstractmethod
    def normalize(self, raw_item: dict, district: str, query: str) -> Optional[PlaceRecord]:
        """Map one raw place dict into a PlaceRecord (or None to skip)."""

    def fetch(self, query: str, district: str, *, refresh: bool = False) -> list[PlaceRecord]:
        # Tiled runs cache under a separate namespace so they never collide with
        # the (60-capped) plain-search cache from earlier runs.
        namespace = getattr(self, "cache_namespace", self.source)
        path = _cache_path(namespace, district, query)
        raw_items = None if refresh else _read_cache(path)
        if raw_items is None:
            raw_items = self.fetch_raw(query, district)
            _write_cache(path, raw_items)
        out: list[PlaceRecord] = []
        for raw in raw_items:
            rec = self.normalize(raw, district, query)
            if rec and rec.place_id:
                out.append(rec)
        return out


# --------------------------------------------------------------------------- #
# Adapter A — Google Places API (New)   [recommended default]
# --------------------------------------------------------------------------- #
class GooglePlacesAdapter(FetchAdapter):
    source = "google"

    def __init__(self, fetch_cfg: dict, api_key: Optional[str] = None):
        self.cfg = fetch_cfg
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")
        self._geocode_cache: dict[tuple[float, float], tuple[str, str]] = {}
        self._bounds_cache: dict[str, tuple[float, float, float, float]] = {}
        tcfg = fetch_cfg.get("tiling", {}) or {}
        self.tiling_enabled = bool(tcfg.get("enabled", False))
        self.tile_grid = int(tcfg.get("grid", 2))            # NxN split per level
        self.tile_max_depth = int(tcfg.get("max_depth", 3))  # recursion levels
        self.tile_cap = int(tcfg.get("cap_threshold", 58))   # >= this => still truncated
        self.search_count = 0                                 # for cost reporting
        if self.tiling_enabled:
            self.cache_namespace = "googletiled"

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=30.0)

    # Statuses worth retrying: rate limits, transient server errors, and the
    # eventually-consistent 403s Google emits right after enabling billing/SKUs.
    _RETRYABLE = {403, 429, 500, 503}
    _MAX_ATTEMPTS = 5

    def _post(self, client: httpx.Client, headers: dict, body: dict, text_query: str) -> dict:
        last_detail = ""
        for attempt in range(self._MAX_ATTEMPTS):
            resp = client.post(GOOGLE_SEARCH_URL, headers=headers, json=body)
            if not resp.is_error:
                return resp.json()
            try:
                last_detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                last_detail = resp.text[:300]
            if resp.status_code in self._RETRYABLE and attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4, 8s
                continue
            raise RuntimeError(
                f"Google Places searchText {resp.status_code} for "
                f"{text_query!r}: {last_detail}"
            )
        raise RuntimeError(
            f"Google Places searchText exhausted retries for {text_query!r}: {last_detail}"
        )

    def _search(
        self,
        client: httpx.Client,
        text_query: str,
        rect: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict]:
        """One paginated searchText, optionally restricted to a (lo_lat, lo_lng,
        hi_lat, hi_lng) rectangle. Returns up to ~60 raw places."""
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": GOOGLE_FIELD_MASK,
        }
        body = {
            "textQuery": text_query,
            "languageCode": self.cfg.get("language_code", "tr"),
            "regionCode": self.cfg.get("region_code", "TR"),
        }
        if rect is not None:
            lo_lat, lo_lng, hi_lat, hi_lng = rect
            body["locationRestriction"] = {
                "rectangle": {
                    "low": {"latitude": lo_lat, "longitude": lo_lng},
                    "high": {"latitude": hi_lat, "longitude": hi_lng},
                }
            }
        places: list[dict] = []
        max_pages = int(self.cfg.get("max_pages", 3))
        for _ in range(max_pages):
            self.search_count += 1
            data = self._post(client, headers, body, text_query)
            places.extend(data.get("places", []))
            token = data.get("nextPageToken")
            if not token:
                break
            body["pageToken"] = token
        return places

    def _district_bounds(self, district: str) -> Optional[tuple[float, float, float, float]]:
        """Authoritative bbox for a district via Geocoding (bounds, else viewport)."""
        if district in self._bounds_cache:
            return self._bounds_cache[district]
        bounds = None
        try:
            with self._client() as client:
                resp = client.get(
                    GOOGLE_GEOCODE_URL,
                    params={"address": f"{district}, {self.cfg.get('city','')}",
                            "key": self.api_key, "language": "tr"},
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    geom = results[0].get("geometry", {})
                    box = geom.get("bounds") or geom.get("viewport")
                    if box:
                        sw, ne = box["southwest"], box["northeast"]
                        bounds = (sw["lat"], sw["lng"], ne["lat"], ne["lng"])
        except Exception:
            bounds = None
        if bounds:
            self._bounds_cache[district] = bounds
        return bounds

    @staticmethod
    def _split_rect(rect: tuple[float, float, float, float], grid: int) -> list[tuple]:
        lo_lat, lo_lng, hi_lat, hi_lng = rect
        dlat = (hi_lat - lo_lat) / grid
        dlng = (hi_lng - lo_lng) / grid
        tiles = []
        for r in range(grid):
            for c in range(grid):
                tiles.append((
                    lo_lat + r * dlat, lo_lng + c * dlng,
                    lo_lat + (r + 1) * dlat, lo_lng + (c + 1) * dlng,
                ))
        return tiles

    def _tile_search(self, client, text_query, rect, depth, acc: dict) -> None:
        """Recursive quad-tree: search a tile; if it still caps and we have depth
        budget, split and recurse. Results merge into acc (dedup by place id)."""
        places = self._search(client, text_query, rect)
        for p in places:
            if p.get("id"):
                acc[p["id"]] = p
        if len(places) >= self.tile_cap and depth < self.tile_max_depth:
            for sub in self._split_rect(rect, self.tile_grid):
                self._tile_search(client, text_query, sub, depth + 1, acc)
                time.sleep(0.02)

    def _tiled_fetch(self, query: str, district: str) -> list[dict]:
        text_query = f"{query} {self.cfg.get('city', '')}".strip()  # district comes from the rectangle
        bounds = self._district_bounds(district)
        acc: dict[str, dict] = {}
        with self._client() as client:
            if bounds is None:
                # Geocode failed — fall back to a plain district text search.
                for p in self._search(client, f"{query} {district} {self.cfg.get('city','')}".strip()):
                    if p.get("id"):
                        acc[p["id"]] = p
            else:
                self._tile_search(client, text_query, bounds, 0, acc)
        return list(acc.values())

    def fetch_raw(self, query: str, district: str) -> list[dict]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
        if self.tiling_enabled:
            return self._tiled_fetch(query, district)
        text_query = f"{query} {district} {self.cfg.get('city', '')}".strip()
        with self._client() as client:
            return self._search(client, text_query)

    def normalize(self, raw: dict, district: str, query: str) -> Optional[PlaceRecord]:
        loc = raw.get("location", {}) or {}
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        components = raw.get("addressComponents", [])

        resolved_district = _pick_component(components, _DISTRICT_TYPES) or district
        neighborhood = _pick_component(components, _NEIGHBORHOOD_TYPES)
        if not neighborhood and lat is not None and lng is not None:
            resolved_district, neighborhood = self._reverse_geocode(lat, lng, resolved_district)

        reviews = []
        for r in (raw.get("reviews") or [])[:3]:
            text = (r.get("text") or {}).get("text") or r.get("originalText", {}).get("text", "")
            if text:
                reviews.append(text.strip())

        return PlaceRecord(
            place_id=raw.get("id", ""),
            name=(raw.get("displayName") or {}).get("text", ""),
            primary_type=raw.get("primaryType", ""),
            type_display=(raw.get("primaryTypeDisplayName") or {}).get("text", ""),
            types=raw.get("types", []) or [],
            address=raw.get("formattedAddress", ""),
            lat=lat,
            lng=lng,
            district=resolved_district,
            neighborhood=neighborhood,
            rating=raw.get("rating"),
            review_count=raw.get("userRatingCount"),
            website=raw.get("websiteUri", "") or "",
            phone=raw.get("nationalPhoneNumber", "") or "",
            business_status=raw.get("businessStatus", "") or "",
            source_queries=[query],
            review_snippets=reviews,
        )

    def _reverse_geocode(self, lat: float, lng: float, fallback_district: str) -> tuple[str, str]:
        key = (round(lat, 5), round(lng, 5))
        if key in self._geocode_cache:
            return self._geocode_cache[key]
        district, neighborhood = fallback_district, ""
        try:
            with self._client() as client:
                resp = client.get(
                    GOOGLE_GEOCODE_URL,
                    params={"latlng": f"{lat},{lng}", "key": self.api_key, "language": "tr"},
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    comps = results[0].get("address_components", [])
                    district = _pick_component(comps, _DISTRICT_TYPES) or fallback_district
                    neighborhood = _pick_component(comps, _NEIGHBORHOOD_TYPES)
        except Exception:
            pass  # best-effort; never let geocoding break the run
        self._geocode_cache[key] = (district, neighborhood)
        return district, neighborhood


# --------------------------------------------------------------------------- #
# Adapter B — Apify Google Maps Scraper
# --------------------------------------------------------------------------- #
class ApifyAdapter(FetchAdapter):
    source = "apify"

    def __init__(self, fetch_cfg: dict, token: Optional[str] = None):
        self.cfg = fetch_cfg
        self.token = token or os.environ.get("APIFY_TOKEN", "")

    def fetch_raw(self, query: str, district: str) -> list[dict]:
        if not self.token:
            raise RuntimeError("APIFY_TOKEN is not set")
        actor = self.cfg.get("apify_actor", "compass/crawler-google-places").replace("/", "~")
        url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
        payload = {
            "searchStringsArray": [query],
            "locationQuery": district,
            "language": self.cfg.get("language_code", "tr"),
            "maxCrawledPlacesPerSearch": int(self.cfg.get("max_results_per_search", 60)),
        }
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, params={"token": self.token}, json=payload)
            resp.raise_for_status()
            return resp.json()

    def normalize(self, raw: dict, district: str, query: str) -> Optional[PlaceRecord]:
        loc = raw.get("location", {}) or {}
        categories = raw.get("categories") or []
        category_name = raw.get("categoryName", "") or ""
        reviews = [
            (r.get("text") or "").strip()
            for r in (raw.get("reviews") or [])[:3]
            if r.get("text")
        ]
        return PlaceRecord(
            place_id=raw.get("placeId", "") or raw.get("fid", ""),
            name=raw.get("title", ""),
            primary_type=category_name,
            type_display=category_name,
            types=categories,
            address=raw.get("address", "") or "",
            lat=loc.get("lat"),
            lng=loc.get("lng"),
            district=raw.get("neighborhood", "") and district or district,
            neighborhood=raw.get("neighborhood", "") or "",
            rating=raw.get("totalScore"),
            review_count=raw.get("reviewsCount"),
            website=raw.get("website", "") or "",
            phone=raw.get("phone", "") or "",
            business_status=raw.get("permanentlyClosed") and "CLOSED_PERMANENTLY" or "OPERATIONAL",
            source_queries=[query],
            review_snippets=reviews,
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_adapter(config: dict) -> FetchAdapter:
    source = config.get("source", "google")
    fetch_cfg = config.get("fetch", {})
    if source == "google":
        return GooglePlacesAdapter(fetch_cfg)
    if source == "apify":
        return ApifyAdapter(fetch_cfg)
    raise ValueError(f"Unknown source: {source!r} (expected 'google' or 'apify')")


def run_fetch(
    config: dict,
    districts: list[str],
    trades: list[dict],
    *,
    refresh: bool = False,
) -> list[PlaceRecord]:
    """Fetch every (district × trade × query_term), dedup by place_id within the run.

    Same place from multiple queries -> keep first, append the extra source_queries.
    """
    adapter = build_adapter(config)
    by_id: dict[str, PlaceRecord] = {}

    for district in districts:
        for trade in trades:
            for term in trade["query_terms"]:
                records = adapter.fetch(term, district, refresh=refresh)
                for rec in records:
                    rec.trade_key = trade["key"]
                    rec.trade_label = trade["label"]
                    existing = by_id.get(rec.place_id)
                    if existing is None:
                        by_id[rec.place_id] = rec
                    else:
                        for q in rec.source_queries:
                            if q not in existing.source_queries:
                                existing.source_queries.append(q)
                # be gentle with the API even when not cached
                time.sleep(0.05)

    if getattr(adapter, "tiling_enabled", False):
        print(f"  (tiling: {adapter.search_count} searchText calls issued this run)")

    return list(by_id.values())
