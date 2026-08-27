"""
Downloads every document listed in data/raw/manifest.json to its local_path,
then updates the manifest with retrieved_date and verified status based on
what actually succeeded. Safe to re-run: skips files already on disk unless
--force is passed.

Usage:
    python ingestion/fetch_corpus.py
    python ingestion/fetch_corpus.py --force
"""

import argparse
import json
import sys
import time
from datetime import date, timezone
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"

HEADERS = {
    # Some Indian govt sites block requests with no/unusual User-Agent.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch_rendered_html(url: str) -> bytes:
    """
    For JS-rendered SPA pages (Parivahan FAQ widgets, incometax.gov.in),
    a plain HTTP GET returns only the page shell. Load it in a real
    (headless) browser instead and grab the DOM after JS has populated it.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, wait_until="networkidle", timeout=45_000)
        # give any post-idle AJAX-populated content a moment to settle
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()
    return html.encode("utf-8")


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def fetch_one(doc: dict, force: bool) -> tuple[bool, str]:
    local_path = ROOT / doc["local_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not force:
        return True, "already on disk, skipped"

    render_js = doc.get("render_js", False)

    last_error = None
    content = None
    for attempt in range(3):
        try:
            if render_js:
                content = fetch_rendered_html(doc["source_url"])
            else:
                resp = requests.get(doc["source_url"], headers=HEADERS, timeout=60)
                resp.raise_for_status()
                content = resp.content
            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(3)
    if last_error is not None:
        return False, f"request failed after 3 attempts: {last_error}"

    if len(content) < 500:
        # Almost certainly an error page / redirect stub, not a real document.
        return False, f"response too small ({len(content)} bytes), likely not the real doc"

    local_path.write_bytes(content)
    return True, f"saved {len(content):,} bytes{' (JS-rendered)' if render_js else ''}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if file already exists")
    args = parser.parse_args()

    manifest = load_manifest()
    today = date.today().isoformat()

    results = []
    for doc in manifest["documents"]:
        ok, msg = fetch_one(doc, args.force)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {doc['doc_id']}: {msg}")
        results.append((doc["doc_id"], ok))

        if ok:
            doc["verified"] = True
            doc["retrieved_date"] = today
        # be polite to government servers
        time.sleep(1)

    save_manifest(manifest)

    failed = [doc_id for doc_id, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} documents fetched successfully.")
    if failed:
        print("Failed (need manual attention - try downloading by hand in a browser):")
        for doc_id in failed:
            print(f"  - {doc_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
