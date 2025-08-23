#!/usr/bin/env python3
"""
Evaluation script for trained Deformable DETR model on chess piece detection.

This script loads a trained model checkpoint and evaluates it on the test set
using COCO-style metrics via pycocotools.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import convert_detr_predictions_to_coco
from train_lightning import ChessDataModule, DeformableDetrLightning
from validation_utils import compute_validation_metrics
from visualization_utils import draw_bbox, setup_matplotlib_backend

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for trained chess piece detection models."""

    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        batch_size: int = 4,
        num_workers: int = 8,
        image_size: int = 256,
    ):
        """
        Initialize model evaluator.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            dataset_root: Root directory of the dataset
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            image_size: Image size for evaluation
            device: Device to run evaluation on
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model and data
        self._load_model()
        self._setup_data()

    def _load_model(self):
        """Load the trained model from checkpoint."""
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")

        # Load the checkpoint
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load model from checkpoint
        self.model = DeformableDetrLightning.load_from_checkpoint(
            self.checkpoint_path, map_location=self.device
        )

        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        logger.info("Model loaded successfully")
        logger.info(
            f"Model has {self.model.num_classes-1} classes"
        )  # -1 for no-object class

    def _setup_data(self):
        """Setup data module and test dataloader."""
        logger.info("Setting up data module...")

        # Create data module
        self.data_module = ChessDataModule(
            dataset_root=self.dataset_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
        )

        # Setup data (this creates train/val/test datasets)
        self.data_module.setup("test")

        # Get test dataloader
        self.test_dataloader = self.data_module.test_dataloader()

        # Get category information
        self.categories = self.data_module.test_dataset.get_categories()

        logger.info(f"Test dataset size: {len(self.data_module.test_dataset)}")
        logger.info(f"Number of test batches: {len(self.test_dataloader)}")
        logger.info(f"Categories: {self.categories}")

    def evaluate(
        self,
        confidence_threshold: float = 0.05,
        save_predictions: bool = False,
        save_images: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Args:
            confidence_threshold: Minimum confidence threshold for predictions
            save_predictions: Whether to save predictions to file
            save_images: Whether to save target and predicted images to disk

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation...")
        start_time = time.time()

        all_predictions = []
        all_targets = []
        all_images_and_predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self.test_dataloader, desc="Evaluating")
            ):
                images, targets, pixel_masks = batch

                # Move to device
                images = images.to(self.device)
                pixel_masks = pixel_masks.to(self.device)

                # Forward pass
                outputs = self.model.model(pixel_values=images, pixel_mask=pixel_masks)

                # Process predictions
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes
                prob = logits.softmax(-1)[..., :-1]  # Remove no-object class
                scores, labels = prob.max(-1)

                batch_predictions = []
                batch_image_data = []
                for i in range(images.size(0)):
                    img_h, img_w = images.shape[2], images.shape[3]
                    boxes = pred_boxes[i]

                    # Convert DETR predictions to COCO format
                    coco_boxes = convert_detr_predictions_to_coco(boxes, img_w, img_h)

                    # Store image predictions for saving
                    if save_images:
                        pred_boxes_list = []
                        pred_scores_list = []
                        pred_labels_list = []

                        for b, s, lab in zip(coco_boxes, scores[i], labels[i]):
                            if s.item() >= confidence_threshold:
                                pred_boxes_list.append(
                                    [b[0].item(), b[1].item(), b[2].item(), b[3].item()]
                                )
                                pred_scores_list.append(s.item())
                                pred_labels_list.append(lab.item())

                        batch_image_data.append(
                            {
                                "image": images[i].cpu(),
                                "target": targets[i],
                                "predictions": {
                                    "boxes": pred_boxes_list,
                                    "scores": pred_scores_list,
                                    "labels": pred_labels_list,
                                },
                                "image_id": int(targets[i]["metadata"]["image_id"]),
                            }
                        )

                    for b, s, lab in zip(coco_boxes, scores[i], labels[i]):
                        if s.item() < confidence_threshold:
                            continue
                        batch_predictions.append(
                            {
                                "image_id": int(targets[i]["metadata"]["image_id"]),
                                "category_id": int(lab.item())
                                + 1,  # COCO format (1-indexed)
                                "bbox": [
                                    b[0].item(),
                                    b[1].item(),
                                    b[2].item(),
                                    b[3].item(),
                                ],
                                "score": s.item(),
                            }
                        )

                all_predictions.extend(batch_predictions)
                all_targets.extend(targets)
                if save_images:
                    all_images_and_predictions.extend(batch_image_data)

                # Log progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"Processed {batch_idx + 1}/{len(self.test_dataloader)} batches"
                    )

        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Total predictions: {len(all_predictions)}")

        # Save predictions if requested
        if save_predictions:
            self._save_predictions(all_predictions, all_targets, confidence_threshold)

        # Save images if requested
        if save_images:
            self._save_images(all_images_and_predictions, confidence_threshold)

        # Compute COCO metrics
        logger.info("Computing COCO metrics...")
        metrics = compute_validation_metrics(
            all_predictions, all_targets, self.categories
        )

        # Add evaluation metadata
        metrics.update(
            {
                "evaluation_time_seconds": evaluation_time,
                "total_predictions": len(all_predictions),
                "total_images": len(all_targets),
                "confidence_threshold": confidence_threshold,
                "checkpoint_path": self.checkpoint_path,
            }
        )

        return metrics

    def _save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
        confidence_threshold: float,
    ):
        """Save predictions and targets to JSON files."""
        # Create output directory
        checkpoint_name = Path(self.checkpoint_path).stem
        output_dir = Path("evaluation_results") / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_file = output_dir / f"predictions_conf_{confidence_threshold}.json"
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to: {pred_file}")

        # Save ground truth (for reference)
        gt_file = output_dir / "ground_truth.json"
        if not gt_file.exists():  # Only save once
            gt_data = []
            for target in targets:
                target_data = {
                    "image_id": int(target["metadata"]["image_id"]),
                    "boxes": target["boxes"].tolist(),
                    "class_labels": target["class_labels"].tolist(),
                    "category_names": target["category_names"],
                    "image_path": target["metadata"]["image_path"],
                    "size": target["metadata"]["size"].tolist(),
                }
                gt_data.append(target_data)

            with open(gt_file, "w") as f:
                json.dump(gt_data, f, indent=2)
            logger.info(f"Ground truth saved to: {gt_file}")

    def _save_images(
        self,
        images_and_predictions: List[Dict[str, Any]],
        confidence_threshold: float,
    ):
        """Save target and predicted images to disk with visualizations."""
        # Create output directory
        checkpoint_name = Path(self.checkpoint_path).stem
        output_dir = Path("evaluation_results") / checkpoint_name / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup matplotlib for saving
        setup_matplotlib_backend("Agg")

        logger.info(
            f"Saving {len(images_and_predictions)} images with visualizations..."
        )

        for idx, data in enumerate(tqdm(images_and_predictions, desc="Saving images")):
            try:
                image = data["image"]
                target = data["target"]
                predictions = data["predictions"]
                image_id = data["image_id"]

                # Create figure with subplots
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                # Unnormalize and convert image to numpy
                img = (
                    self.data_module.test_dataset.unnormalize(image)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]

                # Ground truth visualization (left subplot)
                gt_boxes = target["boxes"].cpu().numpy() if "boxes" in target else []
                gt_names = target.get("category_names", [])

                draw_bbox(axes[0], img.copy(), gt_boxes, gt_names, color="lime")
                axes[0].set_title(f"Ground Truth - Image ID: {image_id}")

                # Predictions visualization (right subplot)
                pred_boxes = predictions["boxes"]
                pred_scores = predictions["scores"]
                pred_labels = predictions["labels"]

                # Convert label IDs to names
                pred_names = []
                for label_id in pred_labels:
                    label_name = self.categories.get(
                        int(label_id), f"class_{int(label_id)}"
                    )
                    pred_names.append(label_name)

                draw_bbox(
                    axes[1],
                    img.copy(),
                    pred_boxes,
                    pred_names,
                    scores=pred_scores,
                    color="red",
                )
                axes[1].set_title(
                    f"Predictions (conf≥{confidence_threshold}) - Image ID: {image_id}"
                )

                plt.tight_layout()

                # Save image
                img_filename = f"image_{image_id:06d}_conf_{confidence_threshold}.png"
                img_path = output_dir / img_filename
                fig.savefig(img_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Log progress every 50 images
                if (idx + 1) % 50 == 0:
                    logger.info(f"Saved {idx + 1}/{len(images_and_predictions)} images")

            except Exception as e:
                logger.error(f"Failed to save image {image_id}: {e}")
                continue

        logger.info(f"Images saved to: {output_dir}")

    def evaluate_multiple_thresholds(
        self, thresholds: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model with multiple confidence thresholds.

        Args:
            thresholds: List of confidence thresholds to evaluate

        Returns:
            Dictionary mapping threshold to metrics
        """
        if thresholds is None:
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

        results = {}
        for threshold in thresholds:
            logger.info(f"Evaluating with confidence threshold: {threshold}")
            metrics = self.evaluate(confidence_threshold=threshold)
            results[f"threshold_{threshold}"] = metrics

            # Log key metrics
            logger.info(
                f"Threshold {threshold}: mAP={metrics['mAP']:.4f}, mAP50={metrics['mAP50']:.4f}"
            )

        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Deformable DETR model"
    )

    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--dataset_root", default="datasets/chessred", help="Root directory of dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size for evaluation (square images)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for predictions",
    )
    parser.add_argument(
        "--save_predictions", action="store_true", help="Save predictions to JSON file"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save target and predicted images to disk",
    )
    parser.add_argument(
        "--multiple_thresholds",
        action="store_true",
        help="Evaluate with multiple confidence thresholds",
    )
    parser.add_argument(
        "--output_file", help="Output file to save metrics (JSON format)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )

        # Run evaluation
        if args.multiple_thresholds:
            results = evaluator.evaluate_multiple_thresholds()

            # Print summary
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS - MULTIPLE THRESHOLDS")
            print("=" * 80)

            for threshold_key, metrics in results.items():
                threshold = threshold_key.replace("threshold_", "")
                print(f"\nThreshold {threshold}:")
                print(f"  mAP     : {metrics['mAP']:.4f}")
                print(f"  mAP@50  : {metrics['mAP50']:.4f}")
                print(f"  mAP@75  : {metrics['mAP75']:.4f}")
                print(f"  AR@100  : {metrics['AR_100']:.4f}")
                print(f"  Predictions: {metrics['total_predictions']}")
                print(f"  Boards 0 mistakes: {metrics['boards_0_mistakes_pct']:.1f}%")
                print(
                    f"  Boards ≤1 mistake: {metrics['boards_1_or_fewer_mistakes_pct']:.1f}%"
                )

        else:
            results = evaluator.evaluate(
                confidence_threshold=args.confidence_threshold,
                save_predictions=args.save_predictions,
                save_images=args.save_images,
            )

            # Print results
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Dataset: {args.dataset_root}")
            print(f"Confidence Threshold: {args.confidence_threshold}")
            print(f"Test Images: {results['total_images']}")
            print(f"Total Predictions: {results['total_predictions']}")
            print(f"Evaluation Time: {results['evaluation_time_seconds']:.2f} seconds")
            print("-" * 80)
            print("COCO Metrics:")
            print(f"  mAP (IoU=0.50:0.95): {results['mAP']:.4f}")
            print(f"  mAP@50 (IoU=0.50)  : {results['mAP50']:.4f}")
            print(f"  mAP@75 (IoU=0.75)  : {results['mAP75']:.4f}")
            print(f"  mAP (small)        : {results['mAP_small']:.4f}")
            print(f"  mAP (medium)       : {results['mAP_medium']:.4f}")
            print(f"  mAP (large)        : {results['mAP_large']:.4f}")
            print("-" * 40)
            print("Average Recall:")
            print(f"  AR@1               : {results['AR_1']:.4f}")
            print(f"  AR@10              : {results['AR_10']:.4f}")
            print(f"  AR@100             : {results['AR_100']:.4f}")
            print(f"  AR (small)         : {results['AR_small']:.4f}")
            print(f"  AR (medium)        : {results['AR_medium']:.4f}")
            print(f"  AR (large)         : {results['AR_large']:.4f}")
            print("-" * 40)
            print("Board-level Metrics:")
            print(
                f"  Boards with 0 mistakes      : {results['boards_0_mistakes_pct']:.1f}%"
            )
            print(
                f"  Boards with ≤1 mistake      : {results['boards_1_or_fewer_mistakes_pct']:.1f}%"
            )
            print(
                f"  Average mistakes per board   : {results['avg_mistakes_per_board']:.2f}"
            )
            print(f"  Total boards analyzed        : {results['total_boards']}")
            print(f"  Total mistakes detected      : {results['total_mistakes']}")
            print("=" * 80)

        # Save results to file if specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
