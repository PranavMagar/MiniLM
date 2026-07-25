"""
scripts/download_minitext8.py
==============================
Downloads the MiniText8 dataset and saves it to data/minitext8/.

MiniText8 is a 100 MB preprocessed subset of enwik9 (Wikipedia XML dump)
containing only lowercase ASCII text.  It is used to validate the full
FinanceLM training pipeline end-to-end.

Official source: http://mattmahoney.net/dc/textdata.html
The file is ``text8.zip`` which contains a single file ``text8``
(100 MB, 100M characters of cleaned Wikipedia text).

Usage
-----
    python scripts/download_minitext8.py
    python scripts/download_minitext8.py --force

Arguments
---------
    --force     Re-download even if the file already exists.
    --log-level Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financelm.paths import MINITEXT8_DIR, MINITEXT8_FILE

# Official text8 download (the standard benchmark used by word2vec, etc.)
_TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_minitext8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the MiniText8 corpus.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file already exists.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    logger.info("Downloading %s", url)
    start = time.perf_counter()

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        data = bytearray()
        downloaded = 0
        chunk = 1 << 16          # 64 KB

        while True:
            block = resp.read(chunk)
            if not block:
                break
            data.extend(block)
            downloaded += len(block)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}%  ({downloaded:,} / {total:,} bytes)",
                      end="", flush=True)

    print()                       # newline after progress
    dest.write_bytes(bytes(data))
    elapsed = time.perf_counter() - start
    logger.info("Downloaded %d bytes in %.1fs → %s", len(data), elapsed, dest)


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("Script      : download_minitext8.py")
    logger.info("Destination : %s", MINITEXT8_FILE)
    logger.info("=" * 60)

    MINITEXT8_DIR.mkdir(parents=True, exist_ok=True)

    if MINITEXT8_FILE.exists() and not args.force:
        logger.info("File already exists — skipping download.")
        logger.info("Use --force to re-download.")
        logger.info("File size : %d bytes", MINITEXT8_FILE.stat().st_size)
        return

    zip_path = MINITEXT8_DIR / "text8.zip"

    # Download zip
    try:
        _download(_TEXT8_URL, zip_path)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    # Validate zip is openable (integrity check without hardcoded hash)
    logger.info("Validating archive …")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"Corrupt file in zip: {bad}")
    except zipfile.BadZipFile as exc:
        logger.error("Archive validation failed: %s", exc)
        zip_path.unlink(missing_ok=True)
        sys.exit(1)
    logger.info("Archive OK")

    # Extract
    logger.info("Extracting …")
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        names = zf.namelist()
        if not names:
            logger.error("ZIP archive is empty.")
            sys.exit(1)
        # The archive contains a single file called 'text8'
        extracted_name = names[0]
        data = zf.read(extracted_name)

    MINITEXT8_FILE.write_bytes(data)
    zip_path.unlink(missing_ok=True)   # remove zip; keep only the text

    logger.info("Saved : %s  (%d bytes)", MINITEXT8_FILE, len(data))
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
