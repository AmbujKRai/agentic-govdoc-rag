"""
Parses every raw file listed in data/raw/manifest.json into clean plain text
under data/processed/<doc_id>.txt, plus a data/processed/<doc_id>.meta.json
sidecar carrying the manifest metadata (doc_type, scenario, source_url,
retrieved_date, etc.) so downstream chunking/embedding never loses provenance.

Usage:
    python ingestion/parse.py
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
PROCESSED_DIR = ROOT / "data" / "processed"

# Tags that are pure site chrome (nav/footer/scripts) on Indian govt sites -
# never contain the actual rules content, always safe to drop.
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]


def clean_html(raw_bytes: bytes) -> str:
    soup = BeautifulSoup(raw_bytes, "lxml")
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse repeated blank lines / whitespace left behind by stripped tags.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    # Each surviving line becomes its own paragraph (blank-line separated) -
    # HTML text has no natural paragraph breaks the way PDF-extracted text
    # does, and chunk.py's paragraph packer needs "\n\n" boundaries to do
    # anything useful with this text.
    return "\n\n".join(lines)


def clean_pdf(local_path: Path) -> str:
    reader = PdfReader(str(local_path))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(f"--- page {i + 1} ---\n{page_text}")
    return "\n\n".join(pages)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_one(doc: dict) -> tuple[bool, str]:
    local_path = ROOT / doc["local_path"]
    if not local_path.exists():
        return False, f"raw file missing at {local_path}, run fetch_corpus.py first"

    suffix = local_path.suffix.lower()
    try:
        if suffix == ".pdf":
            text = clean_pdf(local_path)
        elif suffix in (".html", ".htm"):
            text = clean_html(local_path.read_bytes())
        else:
            return False, f"unsupported file type: {suffix}"
    except Exception as e:
        return False, f"parse error: {e}"

    text = normalize_whitespace(text)
    if len(text) < 200:
        return False, f"suspiciously short output ({len(text)} chars) - check the raw file manually"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_text_path = PROCESSED_DIR / f"{doc['doc_id']}.txt"
    out_meta_path = PROCESSED_DIR / f"{doc['doc_id']}.meta.json"

    out_text_path.write_text(text, encoding="utf-8")
    out_meta_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    return True, f"{len(text):,} chars -> {out_text_path.relative_to(ROOT)}"


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ok_count = 0
    for doc in manifest["documents"]:
        ok, msg = parse_one(doc)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {doc['doc_id']}: {msg}")
        ok_count += ok

    print(f"\n{ok_count}/{len(manifest['documents'])} documents parsed successfully.")


if __name__ == "__main__":
    main()
