"""Loader for model cards / datasheets (markdown package data)."""

from __future__ import annotations

from pathlib import Path

_CARDS_DIR = Path(__file__).resolve().parent / "data" / "modelcards"


def list_cards() -> list[dict]:
    cards = []
    for path in sorted(_CARDS_DIR.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        cards.append({"id": path.stem, "title": title})
    return cards


def get_card(card_id: str) -> str | None:
    path = _CARDS_DIR / f"{card_id}.md"
    if not path.exists() or path.parent != _CARDS_DIR:
        return None
    return path.read_text()
