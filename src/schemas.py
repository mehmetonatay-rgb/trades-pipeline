"""Normalized records that flow between stages.

`PlaceRecord` is the only shape downstream stages know about — both fetch adapters
(Google, Apify) normalize into it. Classification and clustering annotate it in place
via the small dataclasses below.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PlaceRecord:
    place_id: str
    name: str
    primary_type: str = ""
    type_display: str = ""
    types: list[str] = field(default_factory=list)
    address: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    district: str = ""
    neighborhood: str = ""
    rating: Optional[float] = None
    review_count: Optional[int] = None
    website: str = ""
    phone: str = ""
    business_status: str = ""
    # Every query term that surfaced this place (dedup appends extras).
    source_queries: list[str] = field(default_factory=list)
    # Up to a couple of review snippets, used only by the optional LLM pass.
    review_snippets: list[str] = field(default_factory=list)
    # The trade `key` this record was fetched under (e.g. "electrician").
    trade_key: str = ""
    trade_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaceRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Classification:
    label: str          # Service | Supply | Uncertain
    score: float
    reasons: list[str] = field(default_factory=list)
    via_llm: bool = False


@dataclass
class ClassifiedRecord:
    place: PlaceRecord
    classification: Classification

    def to_dict(self) -> dict:
        return {
            "place": self.place.to_dict(),
            "classification": asdict(self.classification),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClassifiedRecord":
        return cls(
            place=PlaceRecord.from_dict(d["place"]),
            classification=Classification(**d["classification"]),
        )


@dataclass
class Cluster:
    route_id: str               # e.g. "Kadıköy-A"
    district: str
    ordered_place_ids: list[str]
    distance_km: float
    maps_urls: list[str]        # one or more (split when > maps_max_waypoints)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Cluster":
        return cls(**d)
