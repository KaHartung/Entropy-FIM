"""
Download PlantUML source files from the-stack-v2 via AWS S3.
"""

import argparse
import logging
import os

import boto3
from smart_open import open
from datasets import load_dataset, Dataset
from huggingface_hub import whoami, login


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("download_stack")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(target_dir: str, debug_download: bool = False) -> None:
    """
    Gather PlantUML code from the-stack-v2 dataset.

    Args:
        target_dir: Directory to save the gathered dataset
        debug_download: Whether to only download a portion of the dataset for quick testing
    """
    save_path = os.path.join(target_dir, "the-stack-v2_PlantUML")
    if os.path.exists(save_path):
        print(f" > Dataset already found at {save_path}. Skipping download!")
        return

    logger = setup_logger()

    # Get AWS credentials from environment
    aws_region = os.getenv("AWS_REGION")
    aws_key = os.getenv("AWS_KEY")
    aws_secret = os.getenv("AWS_SECRET")

    if not all([aws_region, aws_key, aws_secret]):
        raise ValueError("Missing AWS credentials: AWS_REGION, AWS_KEY, AWS_SECRET")

    try:
        login(token=os.getenv("HF_TOKEN"))
        whoami()
    except Exception as e:
        raise EnvironmentError("Not logged in to hf hub!") from e

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    logger.info(" > get hf dataset")
    ds = load_dataset("bigcode/the-stack-v2", "PlantUML", split="train", streaming=True)

    logger.info(" > init session")
    session = boto3.Session(region_name=aws_region,
                            aws_access_key_id=aws_key,
                            aws_secret_access_key=aws_secret)

    s3 = session.client("s3")

    logger.info(" > start download")
    samples = []
    for i, sample in enumerate(ds):
        blob_id = sample["blob_id"]
        encoding = sample["src_encoding"]

        s3_url = f"s3://softwareheritage/content/{blob_id}"

        try:
            with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                content = fin.read().decode(encoding)

            sample["code"] = content
            samples.append(sample)
        except OSError:
            continue
        if debug_download and i >= 1000:
             break

    puml_dataset = Dataset.from_list(samples)
    puml_dataset.save_to_disk(save_path)
    logger.info(f"Saved dataset to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download PlantUML source files from the-stack-v2.")
    parser.add_argument("--target-dir", default="./data/raw", help="Directory to save the downloaded dataset")
    parser.add_argument("--debug-download", action="store_true",
                         help="Only download a portion of the dataset for quick testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.target_dir, args.debug_download)
