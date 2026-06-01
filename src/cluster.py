"""Stage 3 — cluster + route.

DBSCAN (haversine) groups kept leads into route-sized clusters; oversized clusters
are split with a tighter eps; stops are ordered by a nearest-neighbour walk from the
cluster centroid; a Google Maps directions URL is built per cluster (split into
sub-routes when stops exceed the waypoint limit).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from .schemas import ClassifiedRecord, Cluster, PlaceRecord

MAPS_DIR_BASE = "https://www.google.com/maps/dir/?api=1&travelmode=driving&waypoints="


def _haversine_km(a: tuple[float, float], b: tuple[float, float], radius: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def _run_dbscan(coords: np.ndarray, eps_km: float, min_samples: int, radius: float) -> np.ndarray:
    if len(coords) == 0:
        return np.array([], dtype=int)
    eps = eps_km / radius
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")
    return db.fit_predict(np.radians(coords))


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def _nearest_neighbor_order(
    indices: list[int],
    coords: dict[int, tuple[float, float]],
    radius: float,
) -> list[int]:
    """Greedy NN walk starting from the point nearest the cluster centroid."""
    if len(indices) <= 1:
        return list(indices)
    centroid = _centroid([coords[i] for i in indices])
    start = min(indices, key=lambda i: _haversine_km(centroid, coords[i], radius))
    remaining = set(indices)
    remaining.discard(start)
    order = [start]
    current = start
    while remaining:
        nxt = min(remaining, key=lambda i: _haversine_km(coords[current], coords[i], radius))
        order.append(nxt)
        remaining.discard(nxt)
        current = nxt
    return order


def _two_opt(
    order: list[int],
    coords: dict[int, tuple[float, float]],
    radius: float,
) -> list[int]:
    """Improve an open-path tour with 2-opt local search: repeatedly reverse the
    segment between two edges whenever doing so shortens the total path, until no
    improving move remains. The first stop (centroid entry point) is kept fixed.
    Deterministic and dependency-free; typically trims the greedy NN tour by several %.
    """
    n = len(order)
    if n < 4:
        return order  # nothing to untangle on 3 or fewer stops

    def d(a: int, b: int) -> float:
        return _haversine_km(coords[a], coords[b], radius)

    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 2):          # i >= 0 keeps the start fixed
            a, b = best[i], best[i + 1]
            for j in range(i + 2, n):
                c = best[j]
                if j + 1 < n:              # interior edges (a,b) and (c,d)
                    e = best[j + 1]
                    delta = (d(a, c) + d(b, e)) - (d(a, b) + d(c, e))
                else:                      # reversing the suffix: only edge (a,b) changes
                    delta = d(a, c) - d(a, b)
                if delta < -1e-9:
                    best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    a, b = best[i], best[i + 1]
                    improved = True
    return best


def _route_distance_km(order: list[int], coords: dict[int, tuple[float, float]], radius: float) -> float:
    return sum(
        _haversine_km(coords[order[i]], coords[order[i + 1]], radius)
        for i in range(len(order) - 1)
    )


def _maps_urls(order: list[int], coords: dict[int, tuple[float, float]], max_waypoints: int) -> list[str]:
    """Build one or more Maps directions URLs, splitting when stops exceed the limit."""
    urls: list[str] = []
    for start in range(0, len(order), max_waypoints):
        chunk = order[start : start + max_waypoints]
        waypoints = "|".join(f"{coords[i][0]},{coords[i][1]}" for i in chunk)
        urls.append(MAPS_DIR_BASE + waypoints)
    return urls


def _split_oversized(
    members: list[int],
    coords: dict[int, tuple[float, float]],
    cfg: dict,
    radius: float,
) -> list[list[int]]:
    """Re-run DBSCAN on an oversized cluster with a smaller eps to break it up."""
    max_size = cfg["max_cluster_size"]
    if len(members) <= max_size:
        return [members]
    sub_coords = np.array([coords[i] for i in members])
    labels = _run_dbscan(sub_coords, cfg["eps_km"] / 2, max(2, cfg["min_samples"] - 1), radius)
    groups: dict[int, list[int]] = {}
    for member, lbl in zip(members, labels):
        groups.setdefault(int(lbl), []).append(member)
    out: list[list[int]] = []
    for lbl, grp in groups.items():
        # Still too big (or DBSCAN gave up) -> chunk it deterministically.
        if len(grp) > max_size:
            for s in range(0, len(grp), max_size):
                out.append(grp[s : s + max_size])
        else:
            out.append(grp)
    return out


def cluster_leads(kept: list[ClassifiedRecord], config: dict) -> list[Cluster]:
    """Cluster kept leads per district into route-sized groups with ordered stops."""
    cfg = config["cluster"]
    radius = cfg["earth_radius_km"]
    max_waypoints = cfg["maps_max_waypoints"]

    # Index only records that have coordinates.
    records: list[PlaceRecord] = [
        c.place for c in kept if c.place.lat is not None and c.place.lng is not None
    ]
    clusters: list[Cluster] = []

    # Cluster within each district independently (routes don't cross ilçe).
    by_district: dict[str, list[PlaceRecord]] = {}
    for rec in records:
        by_district.setdefault(rec.district or "Bilinmeyen", []).append(rec)

    for district, recs in sorted(by_district.items()):
        coords = {idx: (r.lat, r.lng) for idx, r in enumerate(recs)}
        coord_arr = np.array([coords[i] for i in range(len(recs))])
        labels = _run_dbscan(coord_arr, cfg["eps_km"], cfg["min_samples"], radius)

        groups: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            groups.setdefault(int(lbl), []).append(idx)

        # Noise (-1): attach each to the nearest real cluster centroid; else solo by neighborhood.
        noise = groups.pop(-1, [])
        centroids = {
            lbl: _centroid([coords[i] for i in members]) for lbl, members in groups.items()
        }
        for n in noise:
            if centroids:
                nearest = min(centroids, key=lambda lbl: _haversine_km(coords[n], centroids[lbl], radius))
                groups[nearest].append(n)
            else:
                groups.setdefault(1000 + n, []).append(n)  # solo cluster fallback

        # Split oversized, order, and build routes.
        final_groups: list[list[int]] = []
        for members in groups.values():
            final_groups.extend(_split_oversized(members, coords, cfg, radius))

        for seq, members in enumerate(final_groups):
            # Greedy NN gives a decent starting tour; 2-opt then untangles it.
            order = _nearest_neighbor_order(members, coords, radius)
            order = _two_opt(order, coords, radius)
            route_letter = chr(ord("A") + seq) if seq < 26 else f"Z{seq}"
            route_id = f"{district}-{route_letter}"
            ordered_place_ids = [recs[i].place_id for i in order]
            clusters.append(
                Cluster(
                    route_id=route_id,
                    district=district,
                    ordered_place_ids=ordered_place_ids,
                    distance_km=round(_route_distance_km(order, coords, radius), 3),
                    maps_urls=_maps_urls(order, coords, max_waypoints),
                )
            )

    return clusters


def single_place_maps_url(rec: PlaceRecord) -> str:
    if rec.lat is None or rec.lng is None:
        return ""
    return f"https://www.google.com/maps/search/?api=1&query={rec.lat},{rec.lng}"
