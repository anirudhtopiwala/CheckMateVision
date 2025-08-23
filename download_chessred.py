"""Download and prepare ChessReD dataset.
This script downloads the dataset from 4TU.ResearchData using direct HTTP links derived from the config file at dataset/chessres/dataset.yaml
"""

import os
import zipfile
from pathlib import Path

import click
import wget
import yaml


def progress_bar(current: int, total: int, width: int = 80, name: str = "") -> None:
    """Creates a custom progress bar.


    Args:
        current (int): Current number of downloaded bytes.
        total (int): Total number of bytes.
        width (int, optional): Width of the bar.
        name (str, optional): Name of the object being downloaded.

    Returns:
        None
    """
    file_size_gb = total / (1024**3)
    current_size_gb = current / (1024**3)
    print(
        f"\tDownloading {name}: {int(current / total * 100)}% ",
        f"[{current_size_gb:.2f} / {file_size_gb:.2f}] GB",
        end="\r",
    )


def download_chessred(dataroot: str) -> None:
    """Downloads the ChessReD dataset (idempotent)."""
    with open("datasets/chessred_dataset.yaml", "r") as f:
        chessred_yaml = yaml.safe_load(f)

    print("Checking Chess Recognition Dataset (ChessReD) files...")

    annotation_path = Path(dataroot, "annotations.json")
    img_path = Path(dataroot, "images.zip")

    url_json = chessred_yaml["annotations"]["url"]
    if not annotation_path.exists():
        print("annotations.json not found. Downloading...")
        wget.download(
            url_json,
            annotation_path.as_posix(),
            bar=lambda *args: progress_bar(*args, "annotations"),
        )
        print()  # newline after progress
    else:
        print("annotations.json already exists. Skipping download.")

    url_images = chessred_yaml["images"]["url"]
    if not img_path.exists():
        print("images.zip not found. Downloading...")
        wget.download(
            url_images,
            img_path.as_posix(),
            bar=lambda *args: progress_bar(*args, "images"),
        )
        print("\nImages download completed.")
    else:
        print("images.zip already exists. Skipping download.")

    print("Done.")
    return img_path, annotation_path


def extract_zip(path: str, out_dir: str):
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)


@click.command(help="Download ChessReD dataset assets. Optionally extract the images.")
@click.option(
    "--output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    default="datasets/chessred",
    show_default=True,
    help="Output directory where dataset files will be stored.",
)
@click.option(
    "--extract",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to extract the downloaded images archive.",
)
def main(output: str, extract: bool):
    os.makedirs(output, exist_ok=True)
    # Download dataset if not already present.
    img_zip_path, _ = download_chessred(output)

    if extract:
        extract_dir = os.path.join(output, "images_raw")
        if not os.path.exists(extract_dir):
            print("Extracting images...")
            extract_zip(img_zip_path, extract_dir)
        else:
            print("Images already extracted.")
        print("Now run conversion script to COCO:")
        print(
            "python utils/convert_chessred_to_coco.py --image_dir datasets/chessred/images/train --meta_file datasets/chessred/metadata_train.jsonl --output datasets/chessred/annotations/train.json"
        )


if __name__ == "__main__":
    main()
