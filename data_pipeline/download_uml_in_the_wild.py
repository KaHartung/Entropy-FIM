"""
Download the UML-in-the-wild raw dataset (individual .puml files + metadata) from its
published archive.
"""

import argparse
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Defaults kept here (rather than only in config/data_pipeline.yml) so the pipeline
# still has a sensible source even when config.data_pipeline doesn't override it;
# override via config or --puml-files-zip-url/--metadata-url if this location changes.
DEFAULT_PUML_FILES_ZIP_URL = "https://zenodo.org/records/20537143/files/puml_files.zip"
DEFAULT_METADATA_URL = "https://zenodo.org/records/20537143/files/uml_metadata_enriched.json"


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("download_uml_in_the_wild")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _download(url: str, dest: Path, logger) -> None:
    """Download url to dest, via a same-directory .part file so a killed/failed
    download can never be mistaken for a complete one on a later run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(dest.name + ".part")

    logger.info(f" > downloading {url}")
    last_logged_pct = -1

    def _report(block_count, block_size, total_size):
        nonlocal last_logged_pct
        if total_size <= 0:
            return
        pct = min(100, block_count * block_size * 100 // total_size)
        if pct >= last_logged_pct + 10:
            logger.info(f" >   {pct}%")
            last_logged_pct = pct

    urlretrieve(url, tmp_dest, reporthook=_report)
    tmp_dest.rename(dest)
    logger.info(f" > saved to {dest}")


def _extract_puml_files(zip_path: Path, puml_dir: Path, logger) -> None:
    """
    Extract zip_path's .puml files into puml_dir, flattening away any wrapper
    directory the archive happens to package them under (e.g. every entry nested one
    level deep under a top-level folder, rather than at the archive root) -- so
    puml_dir ends up holding the .puml files directly either way.
    """
    puml_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger.info(f" > extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        puml_files = list(tmp_path.rglob("*.puml"))
        if not puml_files:
            raise RuntimeError(f"No .puml files found inside {zip_path}")

        for puml_file in puml_files:
            shutil.move(str(puml_file), str(puml_dir / puml_file.name))

    logger.info(f" > extracted {len(puml_files)} .puml files to {puml_dir}")


def main(puml_dir: str, metadata_file: str,
         puml_files_zip_url: str = DEFAULT_PUML_FILES_ZIP_URL,
         metadata_url: str = DEFAULT_METADATA_URL) -> None:
    """
    Download the UML-in-the-wild raw dataset: one .puml file per sample (from
    puml_files_zip_url) plus its uml_metadata_enriched.json (from metadata_url).
    Skips whichever half is already present, so re-running after a partial failure
    only fetches what's still missing.

    Args:
        puml_dir: directory to extract the .puml files into
        metadata_file: path to save uml_metadata_enriched.json to
        puml_files_zip_url: URL of the puml_files.zip archive
        metadata_url: URL of uml_metadata_enriched.json
    """
    logger = setup_logger()
    puml_dir_path = Path(puml_dir)
    metadata_path = Path(metadata_file)

    if puml_dir_path.is_dir() and any(puml_dir_path.glob("*.puml")):
        logger.info(f" > .puml files already found in {puml_dir_path}. Skipping download!")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "puml_files.zip"
            _download(puml_files_zip_url, zip_path, logger)
            _extract_puml_files(zip_path, puml_dir_path, logger)

    if metadata_path.exists():
        logger.info(f" > metadata already found at {metadata_path}. Skipping download!")
    else:
        _download(metadata_url, metadata_path, logger)

    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description="Download the UML-in-the-wild raw dataset.")
    parser.add_argument("--puml-dir", default="./data/puml_files", help="Directory to extract the .puml files into")
    parser.add_argument("--metadata-file", default="./data/puml_files/uml_metadata_enriched.json",
                         help="Path to save uml_metadata_enriched.json to")
    parser.add_argument("--puml-files-zip-url", default=DEFAULT_PUML_FILES_ZIP_URL,
                         help="URL of the puml_files.zip archive")
    parser.add_argument("--metadata-url", default=DEFAULT_METADATA_URL,
                         help="URL of uml_metadata_enriched.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        puml_dir=args.puml_dir,
        metadata_file=args.metadata_file,
        puml_files_zip_url=args.puml_files_zip_url,
        metadata_url=args.metadata_url,
    )
