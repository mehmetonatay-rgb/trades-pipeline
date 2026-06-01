"""Stage 2 — classify (THE filter).

Keep people who *do the work*; drop shops that *sell materials*. Three signals
combined into a score (Signal 1 — service-intent query terms — is applied at fetch
time as seeds, so only Signals 2 & 3 score here):

    +allow_category   if primary_type/type_display/types in ALLOW
    +block_category   if ... in BLOCK              (negative weight)
    +service_keyword  per SERVICE name keyword     (capped)
    +supply_keyword   per SUPPLY  name keyword      (capped, negative)
    +no_ecommerce_website  if no website OR website is not an e-commerce/catalog domain

Labels: score >= keep -> Service ; score <= drop -> Supply ; else Uncertain.
An optional LLM pass re-labels the Uncertain band only.
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from .schemas import ClassifiedRecord, Classification, PlaceRecord


def _casefold(s: str) -> str:
    # Turkish-aware-ish lowercasing; str.casefold handles İ/ı reasonably for matching.
    return (s or "").casefold()


def _category_blob(rec: PlaceRecord, generic_types: set[str]) -> list[str]:
    """primary_type + type_display are authoritative; from the noisy `types[]` array
    drop Google's generic catch-all types (store/establishment/…) so they don't
    falsely trigger a block on every listing."""
    blob = [rec.primary_type, rec.type_display]
    blob += [t for t in (rec.types or []) if t not in generic_types]
    return [b for b in blob if b]


def _matches_any(values: list[str], targets: set[str], targets_cf: set[str]) -> bool:
    for v in values:
        if v in targets or _casefold(v) in targets_cf:
            return True
    return False


def _count_keywords(name_cf: str, keywords_cf: list[str]) -> tuple[int, list[str]]:
    hits = [kw for kw in keywords_cf if kw and kw in name_cf]
    return len(hits), hits


def _is_ecommerce(website: str, ecommerce_domains: list[str]) -> bool:
    if not website:
        return False
    host = (urlparse(website).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in ecommerce_domains)


def score_record(rec: PlaceRecord, cfg: dict) -> Classification:
    w = cfg["weights"]
    reasons: list[str] = []
    score = 0.0

    allow = set(cfg.get("allow_categories", []))
    block = set(cfg.get("block_categories", []))
    allow_cf = {_casefold(x) for x in allow}
    block_cf = {_casefold(x) for x in block}
    generic = set(cfg.get("generic_types", []))
    cats = _category_blob(rec, generic)

    # Signal 2 — category allow/block
    if _matches_any(cats, allow, allow_cf):
        score += w["allow_category"]
        reasons.append(f"+{w['allow_category']} allow-category")
    if _matches_any(cats, block, block_cf):
        score += w["block_category"]
        reasons.append(f"{w['block_category']} block-category")

    # Signal 3 — name keyword scoring (capped both directions)
    name_cf = _casefold(rec.name)
    service_cf = [_casefold(k) for k in cfg.get("service_keywords", [])]
    supply_cf = [_casefold(k) for k in cfg.get("supply_keywords", [])]

    n_service, svc_hits = _count_keywords(name_cf, service_cf)
    if n_service:
        contrib = min(n_service * w["service_keyword"], w["service_keyword_cap"])
        score += contrib
        reasons.append(f"+{contrib} service-kw {svc_hits}")

    n_supply, sup_hits = _count_keywords(name_cf, supply_cf)
    if n_supply:
        contrib = max(n_supply * w["supply_keyword"], w["supply_keyword_cap"])
        score += contrib
        reasons.append(f"{contrib} supply-kw {sup_hits}")

    # Website signal — a tradesperson usually has no catalog/e-commerce site.
    if not _is_ecommerce(rec.website, cfg.get("ecommerce_domains", [])):
        score += w["no_ecommerce_website"]
        reasons.append(f"+{w['no_ecommerce_website']} no-ecommerce-website")

    keep = cfg["thresholds"]["keep"]
    drop = cfg["thresholds"]["drop"]
    if score >= keep:
        label = "Service"
    elif score <= drop:
        label = "Supply"
    else:
        label = "Uncertain"

    return Classification(label=label, score=score, reasons=reasons)


# --------------------------------------------------------------------------- #
# Optional LLM pass — Uncertain band only
# --------------------------------------------------------------------------- #
_LLM_PROMPT = (
    "Is this an independent tradesperson/contractor who performs on-site work, "
    "a shop that sells materials/products, or neither?\n"
    "Answer with exactly one word — service, supplier, or other — then a one-line reason.\n\n"
    "Name: {name}\nCategory: {type_display}\nAddress: {address}\nReviews:\n{reviews}"
)


def _llm_classify(rec: PlaceRecord, llm_cfg: dict) -> Optional[tuple[str, str]]:
    """Return (verdict, reason) where verdict ∈ {service, supplier, other}, or None on error."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    snippets = rec.review_snippets[: int(llm_cfg.get("max_review_snippets", 2))]
    prompt = _LLM_PROMPT.format(
        name=rec.name,
        type_display=rec.type_display or rec.primary_type,
        address=rec.address,
        reviews="\n".join(f"- {s}" for s in snippets) or "(none)",
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=llm_cfg.get("model", "claude-haiku-4-5-20251001"),
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text").strip()
    except Exception:
        return None

    lowered = text.lower()
    if lowered.startswith("service"):
        return "service", text
    if lowered.startswith("supplier"):
        return "supplier", text
    return "other", text


def classify_records(
    records: list[PlaceRecord],
    config: dict,
    *,
    use_llm: bool = False,
) -> list[ClassifiedRecord]:
    cfg = config["classify"]
    out: list[ClassifiedRecord] = []
    for rec in records:
        cls = score_record(rec, cfg)
        if use_llm and cls.label == "Uncertain":
            verdict = _llm_classify(rec, cfg.get("llm", {}))
            if verdict is not None:
                v, reason = verdict
                cls.via_llm = True
                cls.reasons.append(f"llm:{reason}")
                if v == "service":
                    cls.label = "Service"
                elif v == "supplier":
                    cls.label = "Supply"
                # "other" -> leave Uncertain for human review
        out.append(ClassifiedRecord(place=rec, classification=cls))
    return out
