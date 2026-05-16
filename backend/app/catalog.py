"""
SHL Product Catalog loader and in-memory store.

Loads the scraped SHL product catalog JSON, normalizes fields,
and provides lookup helpers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Test-type mapping from key categories
# ---------------------------------------------------------------------------
KEY_TO_TYPE_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


@dataclass
class CatalogItem:
    """Normalized representation of a single SHL assessment product."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]
    test_type: str  # comma-separated type codes e.g. "K" or "A,S"
    job_levels: list[str]
    duration: str
    languages: list[str]
    remote: str
    adaptive: str

    # Pre-computed search text (lowercase)
    _search_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            " ".join(self.languages),
        ]
        self._search_text = " ".join(parts).lower()


def _derive_test_type(keys: list[str]) -> str:
    """Map catalog key categories to short type codes."""
    codes: list[str] = []
    for k in keys:
        code = KEY_TO_TYPE_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def _normalize_duration(raw: str) -> str:
    """Extract a clean duration string."""
    if not raw:
        return ""
    m = re.search(r"(\d+)\s*minutes?", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} minutes"
    return raw.strip()


def _summarize_languages(langs: list[str], max_shown: int = 4) -> str:
    """Show first N languages and a count of remaining."""
    if not langs:
        return ""
    if len(langs) <= max_shown:
        return ", ".join(langs)
    shown = ", ".join(langs[:max_shown])
    remaining = len(langs) - max_shown
    return f"{shown} _(+{remaining} more)_"


class Catalog:
    """In-memory SHL product catalog."""

    def __init__(self) -> None:
        self.items: list[CatalogItem] = []
        self._by_id: dict[str, CatalogItem] = {}
        self._by_name_lower: dict[str, CatalogItem] = {}
        self._by_url: dict[str, CatalogItem] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, path: Optional[str] = None) -> None:
        """Load catalog from JSON file."""
        if path is None:
            path = os.environ.get(
                "CATALOG_PATH",
                str(Path(__file__).resolve().parent.parent.parent / "data" / "shl_catalog.json"),
            )
        with open(path, encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = []
        for entry in raw_items:
            if entry.get("status") != "ok":
                continue

            keys = entry.get("keys", [])
            langs = entry.get("languages", [])

            item = CatalogItem(
                entity_id=str(entry.get("entity_id", "")),
                name=entry.get("name", ""),
                url=entry.get("link", ""),
                description=entry.get("description", ""),
                keys=keys,
                test_type=_derive_test_type(keys),
                job_levels=entry.get("job_levels", []),
                duration=_normalize_duration(entry.get("duration", "")),
                languages=langs,
                remote=entry.get("remote", ""),
                adaptive=entry.get("adaptive", ""),
            )
            self.items.append(item)
            self._by_id[item.entity_id] = item
            self._by_name_lower[item.name.lower()] = item
            self._by_url[item.url] = item

        print(f"[catalog] Loaded {len(self.items)} assessments from {path}")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name_lower.get(name.lower())

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        return self._by_url.get(url)

    def search_text(self, item: CatalogItem) -> str:
        return item._search_text

    def to_recommendation_dict(self, item: CatalogItem) -> dict:
        """Convert a CatalogItem to a recommendation dict matching the API schema."""
        return {
            "name": item.name,
            "url": item.url,
            "test_type": item.test_type,
            "keys": ", ".join(item.keys),
            "duration": item.duration or "—",
            "languages": _summarize_languages(item.languages),
        }


# Singleton instance
catalog = Catalog()
