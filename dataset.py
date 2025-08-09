import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

logger = logging.getLogger(__name__)


class ChessPiecesDataset(Dataset):
    """ChessPiecesDataset for loading chess piece annotations in COCO format. It only loads from the chessred2k dataset which have bbox annotations."""

    def __init__(self, dataset_root_dir: str, split: str):
        self.root = dataset_root_dir
        self.split = split

        # Load the annotation file.
        annotation_file_path = os.path.join(self.root, "annotations.json")
        if not os.path.exists(annotation_file_path):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file_path}"
            )
        with open(annotation_file_path, "r") as f:
            logger.info(f"Loading annotations from {annotation_file_path}")
            annotations = json.load(f)

        # Load images, annotations, and categories.
        self.annotations = pd.DataFrame(
            annotations["annotations"]["pieces"], index=None
        )
        self.category_map = {
            cat["id"]: cat["name"] for cat in annotations["categories"]
        }
        self.images = pd.DataFrame(annotations["images"], index=None)

        # Only use chessred2k split as they have bbox annotations.
        self.length = annotations["splits"]["chessred2k"][split]["n_samples"]
        self.split_img_ids = annotations["splits"]["chessred2k"][split]["image_ids"]

        self.annotations = self.annotations[
            self.annotations["image_id"].isin(self.split_img_ids)
        ]
        self.images = self.images[self.images["id"].isin(self.split_img_ids)]

        assert self.length == len(self.split_img_ids) and self.length == len(
            self.images
        ), (
            f"The numeber of images in "
            f"the dataset ({len(self.images)}) for split:{self.split}, does "
            f"not match neither the length specified in the annotations "
            f"({self.length}) or the length of the list of ids for the split "
            f"{len(self.split_img_ids)}"
        )

        # Set the image size and normalization.
        self.image_size = 512
        # Normalization to apply after converting to float [0,1].
        self.mean = torch.tensor([0.47225544, 0.51124555, 0.55296206])
        self.std = torch.tensor([0.27787283, 0.27054584, 0.27802786])
        self.normalize = transforms.Normalize(
            mean=self.mean.tolist(), std=self.std.tolist()
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Unnormalize the image tensor."""
        return img * self.std[:, None, None] + self.mean[:, None, None]

    def get_categories(self) -> dict[int, str]:
        """Return the list of categories."""
        return self.category_map

    def __getitem__(self, idx: int):
        """Return a dataset sample.
        Args:
            idx (int): Index of the sample to return.

        Returns:
            img (Tensor): A 3xHxW Tensor of the resized & normalized image.
            target (Dict): A dictionary containing the resized bounding boxes, labels, and metadata.
        """
        # Get the image.
        image_id = self.split_img_ids[idx]
        img_path = os.path.join(
            self.root,
            "images_raw",
            self.images[self.images["id"] == image_id].path.values[0],
        )
        img = read_image(img_path)  # uint8 tensor CxHxW
        orig_h, orig_w = img.shape[1], img.shape[2]

        # Process the image.
        img = F.resize(img, [self.image_size, self.image_size])
        img = img.float() / 255.0
        img = self.normalize(img)

        # Get the annotations for the image.
        anns = self.annotations[self.annotations["image_id"] == image_id]
        boxes = torch.tensor(
            anns["bbox"].tolist(), dtype=torch.float32
        )  # [N,4] in xywh
        labels = torch.tensor(anns["category_id"].tolist(), dtype=torch.int64)

        # Scale boxes from original size to resized size.
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        if boxes.numel() > 0:
            boxes[:, 0] = boxes[:, 0] * scale_x  # x
            boxes[:, 1] = boxes[:, 1] * scale_y  # y
            boxes[:, 2] = boxes[:, 2] * scale_x  # w
            boxes[:, 3] = boxes[:, 3] * scale_y  # h

        targets = {
            "boxes": boxes,
            "labels": labels,
            "category_names": [self.category_map[label.item()] for label in labels],
            "metadata": {
                "image_id": torch.tensor(image_id, dtype=torch.int64),
                "image_path": img_path,
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
                "size": torch.tensor(
                    [self.image_size, self.image_size], dtype=torch.int64
                ),
                "scales": torch.tensor(
                    [scale_y, scale_x], dtype=torch.float32
                ),  # (y,x)
            },
        }
        return img, targets


def collate_fn(batch):
    """Collate function to handle variable length annotations."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)  # [N, C, H, W]

    # Convert targets to a list of dictionaries.
    targets = [
        {
            "boxes": t["boxes"],
            "labels": t["labels"],
            "category_names": t["category_names"],
            "metadata": t["metadata"],
        }
        for t in targets
    ]

    return images, targets
