import argparse
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset import ChessPiecesDataset
from visualization_utils import setup_matplotlib_backend, draw_bbox

random.seed(42)  # For reproducibility
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ChessPiecesDataset bounding boxes."
    )
    parser.add_argument(
        "--images_root",
        default="datasets/chessred",
        help="Root directory of chessred dataset.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="If set, saves visualizations to this directory instead of showing interactively.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Split to visualize.",
    )
    args = parser.parse_args()

    # Setup matplotlib backend
    if args.save_dir:
        setup_matplotlib_backend("Agg")  # Headless for saving
    else:
        setup_matplotlib_backend("TkAgg")  # Interactive for display

    ds = ChessPiecesDataset(dataset_root_dir=args.images_root, split=args.split)
    indices = random.sample(range(len(ds)), k=args.n)

    images = []
    bbox = []
    category_names = []
    for idx in indices:
        img, target = ds[idx]
        img = ds.unnormalize(img).permute(1, 2, 0).numpy()
        images.append(img)
        bbox.append(target["boxes"].numpy())
        category_names.append(target["category_names"])

    batch_size = 3
    for batch_start in range(0, args.n, batch_size):
        batch_imgs = images[batch_start : batch_start + batch_size]
        batch_boxes = bbox[batch_start : batch_start + batch_size]
        batch_names = category_names[batch_start : batch_start + batch_size]
        fig, axes = plt.subplots(1, len(batch_imgs), figsize=(6 * len(batch_imgs), 6))
        if len(batch_imgs) == 1:
            axes = [axes]
        for ax, img, boxes, names, idx in zip(
            axes,
            batch_imgs,
            batch_boxes,
            batch_names,
            range(batch_start, batch_start + batch_size),
        ):
            draw_bbox(ax, img, boxes, names)
            ax.set_title(f"Sample {idx + 1}")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out_path = os.path.join(
                args.save_dir, f"batch_{batch_start//batch_size + 1}.png"
            )
            fig.savefig(out_path, bbox_inches="tight")
            logger.info(f"Saved image to {out_path}")
        else:
            logger.info(
                f"Showing batch {batch_start // batch_size + 1}. Close the window to exit or continue."
            )
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
