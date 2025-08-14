"""
Visualization utilities for chess piece detection.

This module provides utilities for visualizing bounding boxes on images,
supporting both ground truth annotations and model predictions.
"""

from typing import List, Optional, Dict, Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_bbox(
    ax,
    image: np.ndarray,
    boxes_xywh: List[List[float]],
    label_names: List[str],
    scores: Optional[List[float]] = None,
    color: str = "lime",
    text_color: str = "yellow",
):
    """
    Draw bounding boxes on the image.

    Args:
        ax: Matplotlib axis to draw on
        image: Image array (H, W, C) in range [0, 1] or [0, 255]
        boxes_xywh: List of bounding boxes in [x, y, width, height] format
        label_names: List of class names for each box
        scores: Optional list of confidence scores for each box
        color: Color for bounding box edges
        text_color: Color for text labels
    """
    ax.imshow(image)

    for i, ((x, y, w, h), name) in enumerate(zip(boxes_xywh, label_names)):
        # Draw bounding box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Prepare label text
        if scores is not None and i < len(scores):
            label_text = f"{name}: {scores[i]:.2f}"
        else:
            label_text = name

        # Draw label
        ax.text(
            x,
            y - 2,
            label_text,
            fontsize=8,
            color=text_color,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )
    ax.axis("off")


def setup_matplotlib_backend(backend: str = "Agg") -> None:
    """
    Setup matplotlib backend for different environments.

    Args:
        backend: Matplotlib backend to use ("Agg" for headless, "TkAgg" for interactive)
    """
    import matplotlib

    matplotlib.use(backend)


def visualize_single_image_prediction(
    image: torch.Tensor,
    target: Dict[str, Any],
    predictions: Dict[str, Any],
    category_map: Dict[int, str],
    unnormalize_fn: callable,
    confidence_threshold: float = 0.05,
    title_prefix: str = "",
):
    """
    Create a matplotlib figure showing a single image with targets and predictions side-by-side.
    Returns the figure object for logging to TensorBoard.

    Args:
        image: Single image tensor (C, H, W)
        target: Target dictionary with ground truth for this image
        predictions: Prediction dictionary from model for this image
        category_map: Mapping from category IDs to names
        unnormalize_fn: Function to unnormalize images
        confidence_threshold: Minimum confidence to show predictions
        title_prefix: Prefix for image titles

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create subplot grid: 1 row, 2 columns (GT, Pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Unnormalize and convert image to numpy
    img = unnormalize_fn(image).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]

    # Ground truth visualization (left subplot)
    gt_boxes = target["boxes"].cpu().numpy() if "boxes" in target else []
    gt_names = target.get("category_names", [])

    draw_bbox(axes[0], img.copy(), gt_boxes, gt_names, color="lime")
    axes[0].set_title(f"{title_prefix}Ground Truth")

    # Predictions visualization (right subplot)
    pred_boxes = []
    pred_names = []
    pred_scores = []

    # Extract predictions above confidence threshold
    for box, score, label_id in zip(
        predictions.get("boxes", []),
        predictions.get("scores", []),
        predictions.get("labels", []),
    ):
        if score >= confidence_threshold:
            pred_boxes.append(box)
            pred_scores.append(score)
            # Convert label ID to name (handle 0-indexed vs 1-indexed)
            label_name = category_map.get(int(label_id), f"class_{int(label_id)}")
            pred_names.append(label_name)

    draw_bbox(
        axes[1], img.copy(), pred_boxes, pred_names, scores=pred_scores, color="red"
    )
    axes[1].set_title(f"{title_prefix}Predictions")

    plt.tight_layout()
    return fig
