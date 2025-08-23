"""
Validation utilities for COCO-style evaluation using pycocotools.
"""

import json
import logging
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    COCO-style evaluator for object detection metrics.
    """

    def __init__(self, categories: Dict[int, str]):
        """
        Initialize COCO evaluator.

        Args:
            categories: Dictionary mapping category IDs to category names
                       e.g., {0: 'white_pawn', 1: 'black_pawn', ...}
        """
        self.categories = categories
        self.coco_categories = self._create_coco_categories()

    def _create_coco_categories(self) -> List[Dict[str, Any]]:
        """Create COCO-format category list."""
        coco_categories = []
        for cat_id, cat_name in self.categories.items():
            coco_categories.append(
                {
                    "id": cat_id + 1,  # COCO uses 1-indexed categories
                    "name": cat_name,
                    "supercategory": "chess_piece",
                }
            )
        return coco_categories

    def _create_coco_gt_from_targets(
        self, targets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create COCO-format ground truth annotations from targets.

        Args:
            targets: List of target dictionaries from validation step

        Returns:
            COCO-format ground truth dictionary
        """
        images = []
        annotations = []
        annotation_id = 1
        seen_image_ids = set()

        for target in targets:
            image_id = int(target["metadata"]["image_id"])

            # Add image info only once per image
            if image_id not in seen_image_ids:
                img_size = target["metadata"]["size"]  # [height, width]
                img_h, img_w = int(img_size[0]), int(img_size[1])

                images.append(
                    {
                        "id": image_id,
                        "width": img_w,
                        "height": img_h,
                        "file_name": f"image_{image_id}.jpg",  # Placeholder filename
                    }
                )
                seen_image_ids.add(image_id)

            # Add annotations
            boxes = target["boxes"]  # Should be in COCO format [x, y, w, h]
            labels = target["class_labels"]

            for box, label in zip(boxes, labels):
                # Ensure box is in COCO format [x, y, w, h]
                x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                area = w * h

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(label) + 1,  # COCO uses 1-indexed
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        return {
            "info": {
                "description": "Chess piece detection validation",
                "version": "1.0",
                "year": 2025,
                "contributor": "CheckMateVision",
                "date_created": "2025-01-01",
            },
            "images": images,
            "annotations": annotations,
            "categories": self.coco_categories,
        }

    def evaluate(
        self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate predictions using COCO metrics.

        Args:
            predictions: List of prediction dictionaries in COCO format
            targets: List of target dictionaries from validation step

        Returns:
            Dictionary containing evaluation metrics
        """
        if len(predictions) == 0:
            logger.warning("No predictions to evaluate")
            return {
                "mAP": 0.0,
                "mAP50": 0.0,
                "mAP75": 0.0,
                "mAP_small": 0.0,
                "mAP_medium": 0.0,
                "mAP_large": 0.0,
                "AR_1": 0.0,
                "AR_10": 0.0,
                "AR_100": 0.0,
                "AR_small": 0.0,
                "AR_medium": 0.0,
                "AR_large": 0.0,
            }

        try:
            # Create ground truth COCO object
            gt_dict = self._create_coco_gt_from_targets(targets)

            # Create temporary files for COCO API
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as gt_file:
                json.dump(gt_dict, gt_file)
                gt_file_path = gt_file.name

            # Load ground truth
            coco_gt = COCO(gt_file_path)

            # Load predictions
            coco_dt = coco_gt.loadRes(predictions)

            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics
            metrics = {
                "mAP": float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
                "mAP50": float(coco_eval.stats[1]),  # AP @ IoU=0.50
                "mAP75": float(coco_eval.stats[2]),  # AP @ IoU=0.75
                "mAP_small": float(coco_eval.stats[3]),  # AP @ small objects
                "mAP_medium": float(coco_eval.stats[4]),  # AP @ medium objects
                "mAP_large": float(coco_eval.stats[5]),  # AP @ large objects
                "AR_1": float(coco_eval.stats[6]),  # AR @ maxDets=1
                "AR_10": float(coco_eval.stats[7]),  # AR @ maxDets=10
                "AR_100": float(coco_eval.stats[8]),  # AR @ maxDets=100
                "AR_small": float(coco_eval.stats[9]),  # AR @ small objects
                "AR_medium": float(coco_eval.stats[10]),  # AR @ medium objects
                "AR_large": float(coco_eval.stats[11]),  # AR @ large objects
            }

            # Clean up temporary file
            import os

            os.unlink(gt_file_path)

            logger.info(f"COCO Evaluation Results:")
            logger.info(f"  mAP: {metrics['mAP']:.4f}")
            logger.info(f"  mAP50: {metrics['mAP50']:.4f}")
            logger.info(f"  mAP75: {metrics['mAP75']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during COCO evaluation: {e}")
            # Return zero metrics if evaluation fails
            return {
                "mAP": 0.0,
                "mAP50": 0.0,
                "mAP75": 0.0,
                "mAP_small": 0.0,
                "mAP_medium": 0.0,
                "mAP_large": 0.0,
                "AR_1": 0.0,
                "AR_10": 0.0,
                "AR_100": 0.0,
                "AR_small": 0.0,
                "AR_medium": 0.0,
                "AR_large": 0.0,
            }


def compute_validation_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    categories: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute validation metrics using COCO evaluation and board-level mistake tracking.

    Args:
        predictions: List of prediction dictionaries in COCO format
        targets: List of target dictionaries from validation step
        categories: Dictionary mapping category IDs to names

    Returns:
        Dictionary containing evaluation metrics including board-level metrics
    """
    # Compute standard COCO metrics
    evaluator = COCOEvaluator(categories)
    coco_metrics = evaluator.evaluate(predictions, targets)

    # Compute board-level mistake metrics
    board_metrics = compute_board_level_metrics(predictions, targets, iou_threshold=0.75)

    # Combine all metrics
    all_metrics = {**coco_metrics, **board_metrics}

    return all_metrics


def log_metrics_to_tensorboard(
    metrics: Dict[str, float], logger, global_step: int, prefix: str = "val"
) -> None:
    """
    Log metrics to TensorBoard.

    Args:
        metrics: Dictionary of metrics to log
        logger: Lightning logger instance
        global_step: Current global step
        prefix: Prefix for metric names (e.g., "val", "test")
    """
    if hasattr(logger, "experiment"):
        for metric_name, metric_value in metrics.items():
            logger.experiment.add_scalar(
                f"{prefix}/{metric_name}", metric_value, global_step
            )


def aggregate_predictions_and_targets(
    validation_outputs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Aggregate predictions and targets from validation step outputs.

    Args:
        validation_outputs: List of validation step outputs

    Returns:
        Tuple of (all_predictions, all_targets)
    """
    all_predictions = []
    all_targets = []

    for output in validation_outputs:
        all_predictions.extend(output["predictions"])
        all_targets.extend(output["targets"])

    return all_predictions, all_targets


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between predicted and ground truth boxes.

    Args:
        pred_boxes: Predicted boxes in COCO format [x, y, w, h] (N, 4)
        gt_boxes: Ground truth boxes in COCO format [x, y, w, h] (M, 4)

    Returns:
        IoU matrix of shape (N, M)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    # Convert to [x1, y1, x2, y2] format
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]

    gt_x1 = gt_boxes[:, 0]
    gt_y1 = gt_boxes[:, 1]
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]

    # Compute intersection
    x1 = np.maximum(pred_x1[:, None], gt_x1[None, :])
    y1 = np.maximum(pred_y1[:, None], gt_y1[None, :])
    x2 = np.minimum(pred_x2[:, None], gt_x2[None, :])
    y2 = np.minimum(pred_y2[:, None], gt_y2[None, :])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute areas
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]

    # Compute union
    union = pred_area[:, None] + gt_area[None, :] - intersection

    # Compute IoU
    iou = intersection / (union + 1e-8)
    return iou


def analyze_board_mistakes(
    pred_boxes: List[List[float]],
    pred_labels: List[int],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    gt_labels: List[int],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> int:
    """
    Analyze mistakes on a single board (image).

    Args:
        pred_boxes: Predicted boxes in COCO format [x, y, w, h]
        pred_labels: Predicted class labels (0-indexed)
        pred_scores: Prediction confidence scores
        gt_boxes: Ground truth boxes in COCO format [x, y, w, h]
        gt_labels: Ground truth class labels (0-indexed)
        iou_threshold: IoU threshold for matching (default 0.5)
        score_threshold: Minimum confidence threshold for predictions

    Returns:
        Number of mistakes on this board
    """
    mistakes = 0

    # Filter predictions by score threshold
    valid_pred_indices = [
        i for i, score in enumerate(pred_scores) if score >= score_threshold
    ]
    if not valid_pred_indices:
        # All ground truth detections are missed
        return len(gt_boxes)

    filtered_pred_boxes = [pred_boxes[i] for i in valid_pred_indices]
    filtered_pred_labels = [pred_labels[i] for i in valid_pred_indices]
    filtered_pred_scores = [pred_scores[i] for i in valid_pred_indices]

    if len(gt_boxes) == 0:
        # No ground truth, so no mistakes possible
        return 0

    if len(filtered_pred_boxes) == 0:
        # No valid predictions, all ground truth are missed
        return len(gt_boxes)

    # Convert to numpy arrays
    pred_boxes_np = np.array(filtered_pred_boxes)
    gt_boxes_np = np.array(gt_boxes)

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes_np, gt_boxes_np)

    # Track which ground truth boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    # For each prediction, find the best matching ground truth
    for pred_idx in range(len(filtered_pred_boxes)):
        pred_label = filtered_pred_labels[pred_idx]

        # Find the best IoU match for this prediction
        best_gt_idx = np.argmax(iou_matrix[pred_idx])
        best_iou = iou_matrix[pred_idx, best_gt_idx]

        if best_iou >= iou_threshold:
            # We have a match based on IoU
            gt_label = gt_labels[best_gt_idx]

            if not gt_matched[best_gt_idx]:
                # This GT hasn't been matched yet
                if pred_label != gt_label:
                    # Classification error
                    mistakes += 1
                gt_matched[best_gt_idx] = True
            else:
                # This GT was already matched, so this is a duplicate/false positive
                # We could count this as a mistake, but for simplicity we'll ignore duplicates
                pass
        # If no match found (IoU < threshold), this is a false positive
        # We're not counting false positives as mistakes in this metric

    # Count missed detections (unmatched ground truth boxes)
    missed_detections = np.sum(~gt_matched)
    mistakes += missed_detections

    return mistakes


def compute_board_level_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute board-level mistake metrics.

    Args:
        predictions: List of prediction dictionaries in COCO format
        targets: List of target dictionaries from validation step
        iou_threshold: IoU threshold for matching detections
        score_threshold: Minimum confidence threshold for predictions

    Returns:
        Dictionary containing board-level metrics
    """
    # Group predictions and targets by image_id
    pred_by_image = {}
    gt_by_image = {}

    # Group predictions
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in pred_by_image:
            pred_by_image[image_id] = {"boxes": [], "labels": [], "scores": []}

        # Convert from COCO 1-indexed to 0-indexed labels
        pred_by_image[image_id]["boxes"].append(pred["bbox"])
        pred_by_image[image_id]["labels"].append(pred["category_id"] - 1)
        pred_by_image[image_id]["scores"].append(pred["score"])

    # Group ground truth
    for target in targets:
        image_id = int(target["metadata"]["image_id"])
        if image_id not in gt_by_image:
            gt_by_image[image_id] = {"boxes": [], "labels": []}

        # Targets should already be in the correct format
        for box, label in zip(target["boxes"], target["class_labels"]):
            gt_by_image[image_id]["boxes"].append(
                box.tolist() if hasattr(box, "tolist") else box
            )
            gt_by_image[image_id]["labels"].append(int(label))

    # Get all unique image IDs
    all_image_ids = set(list(pred_by_image.keys()) + list(gt_by_image.keys()))

    boards_with_0_mistakes = 0
    boards_with_1_or_fewer_mistakes = 0
    total_boards = len(all_image_ids)
    total_mistakes = 0

    for image_id in all_image_ids:
        # Get predictions for this image
        pred_data = pred_by_image.get(
            image_id, {"boxes": [], "labels": [], "scores": []}
        )
        gt_data = gt_by_image.get(image_id, {"boxes": [], "labels": []})

        # Analyze mistakes on this board
        mistakes = analyze_board_mistakes(
            pred_boxes=pred_data["boxes"],
            pred_labels=pred_data["labels"],
            pred_scores=pred_data["scores"],
            gt_boxes=gt_data["boxes"],
            gt_labels=gt_data["labels"],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        total_mistakes += mistakes

        if mistakes == 0:
            boards_with_0_mistakes += 1
        if mistakes <= 1:
            boards_with_1_or_fewer_mistakes += 1

    # Compute percentages
    pct_boards_0_mistakes = (
        (boards_with_0_mistakes / total_boards * 100) if total_boards > 0 else 0.0
    )
    pct_boards_1_or_fewer_mistakes = (
        (boards_with_1_or_fewer_mistakes / total_boards * 100)
        if total_boards > 0
        else 0.0
    )
    avg_mistakes_per_board = total_mistakes / total_boards if total_boards > 0 else 0.0

    return {
        "boards_0_mistakes_pct": pct_boards_0_mistakes,
        "boards_1_or_fewer_mistakes_pct": pct_boards_1_or_fewer_mistakes,
        "avg_mistakes_per_board": avg_mistakes_per_board,
        "total_boards": total_boards,
        "total_mistakes": total_mistakes,
    }
