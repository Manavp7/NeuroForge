"""Lightweight retrieval-augmented grounding over a small mechanism corpus.

Builds a TF-IDF index (scikit-learn) over markdown notes in ``data/corpus`` and retrieves the
most relevant passages for a query (e.g., a target name + construct). The agent uses these as
citations so its rationale is grounded in (illustrative) reference text rather than free-form.
No embeddings/API required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass
class Citation:
    doc_id: str
    title: str
    snippet: str
    score: float


class RAGIndex:
    def __init__(self, corpus_dir: Path | None = None):
        self.corpus_dir = corpus_dir or _CORPUS_DIR
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []
        for path in sorted(self.corpus_dir.glob("*.md")):
            text = path.read_text()
            title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
            self.doc_ids.append(path.stem)
            self.titles.append(title)
            self.texts.append(text)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def query(self, text: str, k: int = 2) -> list[Citation]:
        if self.matrix is None or not text.strip():
            return []
        q = self.vectorizer.transform([text])
        sims = linear_kernel(q, self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        out: list[Citation] = []
        for i in order:
            if sims[i] <= 0:
                continue
            body = " ".join(self.texts[i].splitlines()[1:]).strip()
            snippet = (body[:240] + "…") if len(body) > 240 else body
            out.append(Citation(self.doc_ids[i], self.titles[i], snippet, float(round(sims[i], 4))))
        return out


_INDEX: RAGIndex | None = None


def get_index() -> RAGIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = RAGIndex()
    return _INDEX


def cite(query: str, k: int = 2) -> list[dict]:
    return [c.__dict__ for c in get_index().query(query, k=k)]
