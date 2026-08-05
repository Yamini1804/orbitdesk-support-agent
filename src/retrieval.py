"""
Retrieval over the OrbitDesk knowledge base and resolved cases.

Two source types are indexed together but tagged separately:
  - knowledge_base: chunked by markdown ## section, since each KB doc mixes
    several sub-topics (e.g. 03_workspace_settings_and_timezones.md has both
    "Changing the Timezone" and "Other Time-related Behaviour" sections) and
    section-level chunks give tighter, more relevant retrieval than whole-doc.
  - resolved_case: one chunk per case, built from title + symptoms + resolution.
    Cases with status == "superseded" are still indexed (the assignment
    explicitly wants them retrievable for testing) but flagged, so generate.py
    can down-weight them and verify.py can catch a response that treats a
    superseded case as current guidance.

No vector DB - this is ~10 short docs + 8 cases, so an in-memory numpy matrix
with cosine similarity is the right-sized tool. A managed vector database
would be over-engineering for this corpus size.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RESOLVED_CASES_PATH = DATA_DIR / "resolved_cases.json"


@dataclass
class Chunk:
    source_id: str                                    # e.g. "KB-003" or "CASE-1041"
    text: str                                          # embedded text
    passage: str                                       # display/citation text
    source_type: Literal["knowledge_base", "resolved_case"]
    superseded: bool = False


def _extract_document_id(md_text: str, fallback: str) -> str:
    """Pull document_id out of the YAML frontmatter block. Falls back to the
    filename if frontmatter is missing or malformed, so a formatting slip in
    one doc can't silently break the whole index."""
    match = re.search(r"document_id:\s*(\S+)", md_text)
    return match.group(1) if match else fallback


def _split_into_sections(md_text: str) -> List[tuple[str, str]]:
    """Split a KB markdown file into (heading, section_text) pairs on '##'
    headings. Content before the first '##' (the doc intro under the single
    '#' title) is kept as its own section labeled 'Overview'."""
    # Strip YAML frontmatter (--- ... ---) before splitting.
    body = re.sub(r"^---.*?---\s*", "", md_text, flags=re.DOTALL)

    parts = re.split(r"\n##\s+", body)
    sections: List[tuple[str, str]] = []

    first = parts[0].strip()
    first = re.sub(r"^#\s+.*\n?", "", first).strip()  # drop the top-level # title line
    if first:
        sections.append(("Overview", first))

    for part in parts[1:]:
        lines = part.split("\n", 1)
        heading = lines[0].strip()
        text = lines[1].strip() if len(lines) > 1 else ""
        if text:
            sections.append((heading, text))

    return sections


def load_kb_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    for md_path in sorted(KB_DIR.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        doc_id = _extract_document_id(raw, fallback=md_path.stem)

        for heading, section_text in _split_into_sections(raw):
            chunks.append(
                Chunk(
                    source_id=doc_id,
                    text=f"{heading}: {section_text}",
                    passage=section_text,
                    source_type="knowledge_base",
                )
            )
    return chunks


def load_resolved_case_chunks() -> List[Chunk]:
    data = json.loads(RESOLVED_CASES_PATH.read_text(encoding="utf-8"))
    chunks: List[Chunk] = []

    for case in data["cases"]:
        symptoms = " ".join(case.get("symptoms", []))
        resolution = " ".join(case.get("resolution", []))
        text = f"{case['title']}. Symptoms: {symptoms} Resolution: {resolution}"
        passage = (
            f"{case['title']} — Resolution: {resolution}"
            + (f" Limit: {case['important_limit']}" if case.get("important_limit") else "")
        )
        chunks.append(
            Chunk(
                source_id=case["case_id"],
                text=text,
                passage=passage,
                source_type="resolved_case",
                superseded=(case.get("status") == "superseded"),
            )
        )
    return chunks


class Retriever:
    """Loads once, holds embeddings in memory. Instantiate a single instance
    at graph build time (see graph.py) and reuse it across all requests -
    re-embedding the corpus per question would be wasteful and slow."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.chunks: List[Chunk] = load_kb_chunks() + load_resolved_case_chunks()
        corpus_texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(
            corpus_texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 4, min_score: float = 0.25):
        """Returns top_k chunks by cosine similarity (embeddings are
        pre-normalized, so dot product == cosine similarity here).
        min_score filters out weak matches rather than always forcing top_k
        results - a question with genuinely no relevant KB coverage should
        retrieve *nothing*, not the 4 least-bad chunks in the corpus."""
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec

        ranked_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "source_id": chunk.source_id,
                    "passage": chunk.passage,
                    "score": round(score, 4),
                    "source_type": chunk.source_type,
                    "superseded": chunk.superseded,
                }
            )
        return results


if __name__ == "__main__":
    # Quick manual check: run `python -m src.retrieval` from the project root.
    r = Retriever()
    print(f"Indexed {len(r.chunks)} chunks "
          f"({sum(1 for c in r.chunks if c.source_type == 'knowledge_base')} KB, "
          f"{sum(1 for c in r.chunks if c.source_type == 'resolved_case')} resolved cases)\n")

    test_query = "exports stopped after timezone change"
    print(f"Query: {test_query!r}\n")
    for hit in r.search(test_query):
        flag = " [SUPERSEDED]" if hit["superseded"] else ""
        print(f"  [{hit['score']}] {hit['source_id']}{flag}: {hit['passage'][:100]}...")
