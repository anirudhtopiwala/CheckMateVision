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
    Compute validation metrics using COCO evaluation.

    Args:
        predictions: List of prediction dictionaries in COCO format
        targets: List of target dictionaries from validation step
        categories: Dictionary mapping category IDs to names

    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = COCOEvaluator(categories)
    return evaluator.evaluate(predictions, targets)


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
