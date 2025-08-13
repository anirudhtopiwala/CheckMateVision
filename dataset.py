import json
import logging
import os

import albumentations as A
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.ops import box_convert

logger = logging.getLogger(__name__)


def convert_boxes_to_detr_format(
    boxes: torch.Tensor, img_width: int, img_height: int
) -> torch.Tensor:
    """
    Convert bounding boxes from COCO format (xywh) to normalized cxcywh format for DETR models.

    Args:
        boxes: Tensor of shape [N, 4] in COCO format (x, y, width, height)
        img_width: Image width for normalization
        img_height: Image height for normalization

    Returns:
        Tensor of shape [N, 4] in normalized cxcywh format
    """
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=boxes.device)

    # Convert xywh to cxcywh using box_convert
    cxcywh_boxes = box_convert(boxes, in_fmt="xywh", out_fmt="cxcywh")

    # Normalize by image dimensions
    normalized_boxes = cxcywh_boxes / torch.tensor(
        [img_width, img_height, img_width, img_height],
        dtype=torch.float32,
        device=boxes.device,
    )

    return normalized_boxes


def convert_detr_predictions_to_coco(
    pred_boxes: torch.Tensor, img_width: int, img_height: int
) -> torch.Tensor:
    """
    Convert DETR model predictions from normalized cxcywh to COCO format (xywh).

    Args:
        pred_boxes: Tensor of shape [N, 4] in normalized cxcywh format
        img_width: Image width for denormalization
        img_height: Image height for denormalization

    Returns:
        Tensor of shape [N, 4] in COCO format (x, y, width, height)
    """
    if pred_boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=pred_boxes.device)

    # Denormalize to pixel coordinates
    pixel_boxes = pred_boxes * torch.tensor(
        [img_width, img_height, img_width, img_height],
        dtype=torch.float32,
        device=pred_boxes.device,
    )

    # Convert cxcywh to xywh using box_convert
    coco_boxes = box_convert(pixel_boxes, in_fmt="cxcywh", out_fmt="xywh")

    return coco_boxes


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

        # Set the image size and normalization to match Deformable DETR expectations.
        # Use ImageNet normalization values that the pretrained model expects
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet mean
        self.std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet std
        self.normalize = transforms.Normalize(
            mean=self.mean.tolist(), std=self.std.tolist()
        )

        # Deformable DETR expects: shortest_edge=800, longest_edge=1333
        self.shortest_edge = 256
        self.longest_edge = 256

        # Data augmentations for training
        self.is_train = split == "train"
        if self.is_train:
            self.augmentations = A.Compose(
                [
                    A.Rotate(limit=10, p=0.5),  # random rotation ±10°
                    A.RandomScale(scale_limit=0.1, p=0.5),  # scaling
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                    ),  # color jitter
                    A.GaussNoise(
                        std_range=(0.012, 0.028),
                        mean_range=(0, 0),
                        p=0.3,
                    ),
                    A.HorizontalFlip(p=0.5),  # horizontal flip
                    A.SmallestMaxSize(
                        max_size=self.shortest_edge, p=1.0
                    ),  # Resize shortest edge to 800
                    A.LongestMaxSize(
                        max_size=self.longest_edge, p=1.0
                    ),  # Limit longest edge to 1333
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    label_fields=["category_ids"],
                    min_visibility=0.1,  # Keep boxes with at least 10% visibility
                ),
            )
        else:
            # For validation/test, just resize maintaining aspect ratio
            self.augmentations = A.Compose(
                [
                    A.SmallestMaxSize(max_size=self.shortest_edge, p=1.0),
                    A.LongestMaxSize(max_size=self.longest_edge, p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    label_fields=["category_ids"],
                ),
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Unnormalize the image tensor."""
        # Move mean and std to the same device as the image for operations
        mean = self.mean.to(img.device)
        std = self.std.to(img.device)
        return img * std[:, None, None] + mean[:, None, None]

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

        # Get the annotations for the image.
        anns = self.annotations[self.annotations["image_id"] == image_id]
        boxes = torch.tensor(
            anns["bbox"].tolist(), dtype=torch.float32
        )  # [N,4] in xywh - keep on CPU initially
        labels = torch.tensor(
            anns["category_id"].tolist(), dtype=torch.int64
        )  # keep on CPU initially

        # Convert to numpy for albumentations.
        img_np = img.permute(1, 2, 0).numpy()  # HWC

        # Apply albumentations (including resize and augmentations)
        if boxes.numel() > 0:
            try:
                transformed = self.augmentations(
                    image=img_np,
                    bboxes=boxes.cpu().numpy(),
                    category_ids=labels.cpu().tolist(),
                )
                img_np = transformed["image"]
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["category_ids"], dtype=torch.int64)
            except Exception as e:
                print(f"Augmentation failed for image {image_id}: {e}")
                resize_only = A.Compose(
                    [
                        A.SmallestMaxSize(max_size=self.shortest_edge, p=1.0),
                        A.LongestMaxSize(max_size=self.longest_edge, p=1.0),
                    ],
                    bbox_params=A.BboxParams(
                        format="coco", label_fields=["category_ids"]
                    ),
                )
                transformed = resize_only(
                    image=img_np,
                    bboxes=boxes.cpu().numpy(),
                    category_ids=labels.cpu().tolist(),
                )
                img_np = transformed["image"]
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["category_ids"], dtype=torch.int64)
        else:
            # No boxes, just resize the image
            resize_only = A.Compose(
                [
                    A.SmallestMaxSize(max_size=self.shortest_edge, p=1.0),
                    A.LongestMaxSize(max_size=self.longest_edge, p=1.0),
                ]
            )
            img_np = resize_only(image=img_np)["image"]

        # Convert back to tensor
        img = torch.from_numpy(img_np).permute(2, 0, 1)  # CHW

        # Calculate scales for metadata (based on final size vs original)
        final_h, final_w = img_np.shape[:2]
        scale_x = final_w / orig_w
        scale_y = final_h / orig_h

        # Normalize
        img = img.float() / 255.0
        img = self.normalize(img)

        targets = {
            "boxes": boxes,
            "class_labels": labels,
            "category_names": (
                [self.category_map[label.item()] for label in labels]
                if labels.numel() > 0
                else []
            ),
            "metadata": {
                "image_id": torch.tensor(image_id, dtype=torch.int64),
                "image_path": img_path,
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
                "size": torch.tensor([final_h, final_w], dtype=torch.int64),
                "scales": torch.tensor(
                    [scale_y, scale_x], dtype=torch.float32
                ),  # (y,x)
            },
        }
        return img, targets


def collate_fn(batch):
    """Collate function to handle variable length annotations and box format conversions."""
    images, targets = zip(*batch)

    # Find maximum dimensions in the batch
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # Pad images and create pixel masks
    padded_images = []
    pixel_masks = []

    for img in images:
        C, H, W = img.shape

        # Create padded image with zeros
        padded_img = torch.zeros((C, max_height, max_width), dtype=img.dtype)
        padded_img[:, :H, :W] = img
        padded_images.append(padded_img)

        # Create pixel mask (True for valid pixels, False for padding)
        pixel_mask = torch.zeros((max_height, max_width), dtype=torch.bool)
        pixel_mask[:H, :W] = True
        pixel_masks.append(pixel_mask)

    images = torch.stack(padded_images, dim=0)  # [N, C, H, W]
    pixel_masks = torch.stack(pixel_masks, dim=0)  # [N, H, W]

    # Use max dimensions for normalization
    H, W = max_height, max_width

    collated_targets = []
    for t in targets:
        # Ensure boxes and labels are valid tensors
        if t["boxes"].numel() == 0:
            # Create empty tensors with correct shape for images with no annotations
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            category_names = []
            # Empty normalized boxes for HuggingFace format
            normalized_boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes = t["boxes"]  # COCO format (xywh)
            labels = t["class_labels"]
            category_names = t["category_names"]

            # Convert COCO format to normalized cxcywh for HuggingFace DETR using utility function
            normalized_boxes = convert_boxes_to_detr_format(boxes, W, H)

        collated_targets.append(
            {
                "boxes": boxes,  # Original COCO format for compatibility
                "normalized_boxes": normalized_boxes,  # Normalized cxcywh for HuggingFace DETR
                "class_labels": labels,
                "category_names": category_names,
                "metadata": t["metadata"],
            }
        )

    return images, collated_targets, pixel_masks
