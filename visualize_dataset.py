import argparse
import os
import random
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from dataset import ChessPiecesDataset

random.seed(42)  # For reproducibility
import matplotlib

matplotlib.use("TkAgg")
import logging

logger = logging.getLogger(__name__)


def draw_bbox(
    ax, image: np.array, boxes_xywh: List[List[float]], label_names: List[str]
):
    """Draw bounding boxes on the image."""
    ax.imshow(image)
    for (x, y, w, h), name in zip(boxes_xywh, label_names):
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 2,
            name,
            fontsize=8,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )
    ax.axis("off")


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
