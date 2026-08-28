"""
Splits each cleaned document in data/processed/*.txt into overlapping,
paragraph-aware chunks and writes them all to data/processed/chunks.jsonl -
one JSON object per line, each carrying full provenance (doc_id, doc_type,
scenario, source_url, retrieved_date, page) so retrieval can always cite
back to an exact source.

Chunking strategy: pack whole paragraphs into a chunk up to MAX_CHARS: never
split a paragraph mid-sentence. Carry the last paragraph of each chunk into
the next one as overlap, so a fact split across a paragraph boundary is
still fully present in at least one chunk.

Usage:
    python ingestion/chunk.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

MAX_CHARS = 1000
MIN_CHARS = 100  # drop trailing scraps smaller than this (usually nav/footer debris)

PAGE_MARKER_RE = re.compile(r"^--- page (\d+) ---$")


def split_paragraphs_with_pages(text: str) -> list[tuple[str, int | None]]:
    """Returns [(paragraph_text, page_number_or_None), ...], tracking the
    most recent '--- page N ---' marker (present in PDF-derived text)."""
    current_page = None
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = PAGE_MARKER_RE.match(para)
        if m:
            current_page = int(m.group(1))
            continue
        out.append((para, current_page))
    return out


def split_oversized_paragraph(para: str, max_chars: int) -> list[str]:
    """A 'paragraph' that itself exceeds max_chars (PDF tables extract as one
    blank-line-free blob with no paragraph breaks between rows - same root
    cause as the HTML-without-blank-lines issue parse.py fixes) needs to be
    split further. Pack it line-by-line, and hard-slice any single line that
    is still too long on its own (defensive - shouldn't normally happen)."""
    if len(para) <= max_chars:
        return [para]

    pieces = []
    current_lines: list[str] = []
    current_len = 0
    for line in para.split("\n"):
        if len(line) > max_chars:
            if current_lines:
                pieces.append("\n".join(current_lines))
                current_lines, current_len = [], 0
            for i in range(0, len(line), max_chars):
                pieces.append(line[i:i + max_chars])
            continue

        if current_len + len(line) > max_chars and current_lines:
            pieces.append("\n".join(current_lines))
            current_lines, current_len = [line], len(line)
        else:
            current_lines.append(line)
            current_len += len(line)

    if current_lines:
        pieces.append("\n".join(current_lines))
    return pieces


def expand_oversized_paragraphs(
    paragraphs: list[tuple[str, int | None]], max_chars: int
) -> list[tuple[str, int | None]]:
    out = []
    for para, page in paragraphs:
        for piece in split_oversized_paragraph(para, max_chars):
            out.append((piece, page))
    return out


def pack_chunks(paragraphs: list[tuple[str, int | None]]) -> list[dict]:
    chunks = []
    current_paras: list[str] = []
    current_len = 0
    current_start_page = None

    def flush():
        nonlocal current_paras, current_len, current_start_page
        if current_paras and current_len >= MIN_CHARS:
            chunks.append({
                "text": "\n\n".join(current_paras),
                "start_page": current_start_page,
            })

    for para, page in paragraphs:
        if current_start_page is None:
            current_start_page = page

        if current_len + len(para) > MAX_CHARS and current_paras:
            flush()
            # overlap: carry the last paragraph forward into the next chunk
            overlap_para = current_paras[-1]
            current_paras = [overlap_para, para]
            current_len = len(overlap_para) + len(para)
            current_start_page = page
        else:
            current_paras.append(para)
            current_len += len(para)

    flush()
    return chunks


def main():
    meta_files = sorted(PROCESSED_DIR.glob("*.meta.json"))
    if not meta_files:
        print("No *.meta.json files found in data/processed/ - run parse.py first.")
        return

    all_chunks = []
    for meta_path in meta_files:
        doc_id = meta_path.stem.replace(".meta", "")
        text_path = PROCESSED_DIR / f"{doc_id}.txt"
        if not text_path.exists():
            print(f"[SKIP] {doc_id}: missing {text_path.name}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = text_path.read_text(encoding="utf-8")

        paragraphs = split_paragraphs_with_pages(text)
        paragraphs = expand_oversized_paragraphs(paragraphs, MAX_CHARS)
        packed = pack_chunks(paragraphs)

        for i, c in enumerate(packed):
            all_chunks.append({
                "chunk_id": f"{doc_id}::{i}",
                "doc_id": doc_id,
                "doc_type": meta["doc_type"],
                "scenario": meta["scenario"],
                "title": meta["title"],
                "source_url": meta["source_url"],
                "retrieved_date": meta.get("retrieved_date"),
                "page": c["start_page"],
                "text": c["text"],
            })

        print(f"[OK] {doc_id}: {len(packed)} chunks")

    CHUNKS_PATH.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks),
        encoding="utf-8",
    )
    print(f"\nWrote {len(all_chunks)} total chunks -> {CHUNKS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
