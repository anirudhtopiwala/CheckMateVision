import argparse
import io
import logging
import os
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

from dataset import ChessPiecesDataset, collate_fn, convert_detr_predictions_to_coco
from visualization_utils import (
    setup_matplotlib_backend,
    visualize_single_image_prediction,
)
from validation_utils import (
    compute_validation_metrics,
    log_metrics_to_tensorboard,
    aggregate_predictions_and_targets,
)

logger = logging.getLogger(__name__)


class DeformableDetrLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 12,
        num_queries: int = 32,
        lr_backbone: float = 1e-5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_iters: int = 1000,
        max_iters: int = None,
        pretrained: bool = True,
        visualize_every_n_steps: int = 150,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes + 1  # +1 for no-object class in COCO format

        # Initialize validation outputs storage
        self.validation_step_outputs = []

        # Visualization parameters
        self.visualize_every_n_steps = visualize_every_n_steps
        self.visualization_confidence_threshold = 0.1

        # Validation step counter to track validation steps across epochs
        self.validation_step_count = 0

        # Build model with updated queries and num_classes
        if pretrained:
            self.model = DeformableDetrForObjectDetection.from_pretrained(
                "SenseTime/deformable-detr",
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
            # Update number of queries
            self.model.config.num_queries = num_queries
            # Reinitialize query embeddings with new size
            hidden_dim = self.model.config.d_model
            # The query embeddings need to be twice the hidden dimension
            # because they get split into query_embed and target
            self.model.model.query_position_embeddings = torch.nn.Embedding(
                num_queries, hidden_dim * 2
            )
        else:
            config = DeformableDetrConfig(
                num_labels=self.num_classes,
                num_queries=num_queries,
            )
            self.model = DeformableDetrForObjectDetection(config)

        self.lr_backbone = lr_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters

        # Freeze backbone initially (will be trained with lower LR)
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Backbone will be trained with lower learning rate via parameter groups"""
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = True  # Keep trainable but will use lower LR

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        images, targets, pixel_masks = batch

        labels = []
        for t in targets:
            labels.append(
                {
                    "class_labels": t["class_labels"].to(self.device),
                    "boxes": t["normalized_boxes"].to(self.device),
                }
            )

        # Forward pass
        outputs = self(
            pixel_values=images.to(self.device),
            pixel_mask=pixel_masks.to(self.device),
            labels=labels,
        )
        loss = outputs.loss

        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Trigger visualization every N steps for training data
        if (self.global_step + 1) % self.visualize_every_n_steps == 0:
            self.visualize_predictions(
                images=images,
                targets=targets,
                pixel_masks=pixel_masks,
                batch_idx=0,
                step=self.global_step,
                mode="train",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, pixel_masks = batch

        # Prepare labels for validation loss computation
        labels = []
        for t in targets:
            labels.append(
                {
                    "class_labels": t["class_labels"].to(self.device),
                    "boxes": t["normalized_boxes"].to(self.device),
                }
            )

        with torch.no_grad():
            outputs = self(
                pixel_values=images.to(self.device),
                pixel_mask=pixel_masks.to(self.device),
                labels=labels,
            )

        # Compute and log validation loss
        val_loss = outputs.loss
        self.log(
            "loss/val",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Trigger visualization every N validation batches (only for first batch)
        if (
            self.validation_step_count + 1
        ) % self.visualize_every_n_steps == 0:
            self.visualize_predictions(
                images=images,
                targets=targets,
                pixel_masks=pixel_masks,
                batch_idx=0,
                step=self.validation_step_count,
                mode="val",
            )

        # Increment validation step counter
        self.validation_step_count += 1

        # Store predictions for epoch-end evaluation
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        prob = logits.softmax(-1)[..., :-1]  # Remove no-object class
        scores, labels = prob.max(-1)

        predictions = []
        for i in range(images.size(0)):
            img_h, img_w = images.shape[2], images.shape[3]
            boxes = pred_boxes[i]

            # Convert DETR predictions to COCO format using utility function
            coco_boxes = convert_detr_predictions_to_coco(boxes, img_w, img_h)
            for b, s, lab in zip(coco_boxes, scores[i], labels[i]):
                if s.item() < 0.05:  # Confidence threshold
                    continue
                predictions.append(
                    {
                        "image_id": int(targets[i]["metadata"]["image_id"]),
                        "category_id": int(lab.item()) + 1,  # COCO format (1-indexed)
                        "bbox": [b[0].item(), b[1].item(), b[2].item(), b[3].item()],
                        "score": s.item(),
                    }
                )

        step_output = {"predictions": predictions, "targets": targets}
        self.validation_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self):
        # Aggregate all predictions and targets
        all_predictions, all_targets = aggregate_predictions_and_targets(
            self.validation_step_outputs
        )

        if len(all_predictions) == 0:
            # Log zero metrics if no predictions
            metrics = {
                "val_mAP": 0.0,
                "val_mAP50": 0.0,
                "val_mAP75": 0.0,
                "val_AR_100": 0.0,
            }
            for metric_name, metric_value in metrics.items():
                self.log(metric_name, metric_value, sync_dist=True)
        else:
            # Get category map from the dataset
            category_map = self.trainer.datamodule.val_dataset.get_categories()

            # Compute COCO metrics
            metrics = compute_validation_metrics(
                all_predictions, all_targets, category_map
            )

            # Log main metrics to Lightning
            self.log("val_mAP", metrics["mAP"], sync_dist=True, prog_bar=True)
            self.log("val_mAP50", metrics["mAP50"], sync_dist=True, prog_bar=True)
            self.log("val_mAP75", metrics["mAP75"], sync_dist=True)
            self.log("val_AR_100", metrics["AR_100"], sync_dist=True)

            # Log all metrics to TensorBoard
            if hasattr(self.logger, "experiment"):
                log_metrics_to_tensorboard(
                    metrics, self.logger, global_step=self.global_step, prefix="val"
                )

            # Log summary
            self.print(
                f"Validation Metrics - mAP: {metrics['mAP']:.4f}, mAP50: {metrics['mAP50']:.4f}"
            )

        # Clear the outputs for next epoch
        self.validation_step_outputs.clear()

    def visualize_predictions(
        self, images, targets, pixel_masks, batch_idx, step, mode
    ):
        """Generate visualization of predictions vs ground truth and log to TensorBoard."""
        try:
            setup_matplotlib_backend("Agg")  # Use non-interactive backend
            # Get model predictions
            with torch.no_grad():
                outputs = self(pixel_values=images, pixel_mask=pixel_masks)

            # Process predictions for the single image
            logits = outputs.logits
            pred_boxes = outputs.pred_boxes
            prob = logits.softmax(-1)[..., :-1]  # Remove no-object class
            scores, labels = prob.max(-1)

            # Process the single image
            img_h, img_w = images.shape[-2], images.shape[-1]
            boxes = pred_boxes[batch_idx]
            scores_img = scores[batch_idx]
            labels_img = labels[batch_idx]

            # Convert DETR predictions to COCO format
            coco_boxes = convert_detr_predictions_to_coco(boxes, img_w, img_h)

            # Filter by confidence and prepare for visualization
            pred_boxes_list = []
            pred_scores_list = []
            pred_labels_list = []

            for box, score, label in zip(coco_boxes, scores_img, labels_img):
                if score.item() >= self.visualization_confidence_threshold:
                    pred_boxes_list.append(
                        [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
                    )
                    pred_scores_list.append(score.item())
                    pred_labels_list.append(label.item())

            predictions = {
                "boxes": pred_boxes_list,
                "scores": pred_scores_list,
                "labels": pred_labels_list,
            }

            # Get category map from trainer's datamodule
            category_map = self.trainer.datamodule.train_dataset.get_categories()
            unnormalize_fn = self.trainer.datamodule.train_dataset.unnormalize

            # Create visualization figure for single image
            fig = visualize_single_image_prediction(
                image=images[batch_idx],
                target=targets[batch_idx],
                predictions=predictions,
                category_map=category_map,
                unnormalize_fn=unnormalize_fn,
                confidence_threshold=self.visualization_confidence_threshold,
                title_prefix=f"Step {self.global_step} - ",
            )

            # Convert figure to image tensor for TensorBoard
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)

            # Convert to PIL Image and then to tensor
            pil_img = Image.open(buf)
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(pil_img)

            # Log to TensorBoard with mode-specific tag
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.add_image(
                    f"{mode}_predictions", img_tensor, global_step=step
                )

            # Close the figure and buffer
            plt.close(fig)
            buf.close()

            self.print(f"Logged {mode} visualization to TensorBoard at step {step}")

        except Exception as e:
            self.print(f"Failed to generate {mode} visualization: {e}")
            traceback.print_exc()

    def configure_optimizers(self):
        # Separate parameter groups for backbone and other components
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": other_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        if self.max_iters is None:
            return optimizer

        # Cosine annealing with warmup
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_iters,
            num_training_steps=self.max_iters,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class ChessDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 2,
        num_workers: int = 8,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ChessPiecesDataset(
                dataset_root_dir=self.dataset_root, split="train"
            )
            self.val_dataset = ChessPiecesDataset(
                dataset_root_dir=self.dataset_root, split="val"
            )

        if stage == "test" or stage is None:
            self.test_dataset = ChessPiecesDataset(
                dataset_root_dir=self.dataset_root, split="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )


def train(args):
    # Log output directory.
    print(f"Output directory: {args.output_dir}")

    # Data module
    data_module = ChessDataModule(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Calculate max iterations for scheduler
    data_module.setup("fit")
    steps_per_epoch = len(data_module.train_dataloader())
    max_iters = steps_per_epoch * args.epochs

    category_map = data_module.train_dataset.get_categories()
    num_classes = len(category_map)

    # Model
    model = DeformableDetrLightning(
        num_classes=num_classes,
        num_queries=32,  # Maximum chess pieces on board
        lr_backbone=args.lr_backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        max_iters=max_iters,
        visualize_every_n_steps=args.visualize_every_n_steps,
    )

    # Callbacks
    # Save best checkpoint based on validation mAP
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_mAP50:.3f}",
        monitor="val_mAP50",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # Save checkpoint at the end of every epoch
    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints", "epochs"),
        filename="epoch-{epoch:02d}-{loss/val:.4f}",
        save_top_k=-1,
        every_n_epochs=10,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="deformable_detr_chess",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        logger=tensorboard_logger,
        callbacks=[best_checkpoint_callback, epoch_checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Training
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )

    print(f"Best checkpoint saved at: {best_checkpoint_callback.best_model_path}")
    print(
        f"Epoch checkpoints saved in: {os.path.join(args.output_dir, 'checkpoints', 'epochs')}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Deformable DETR with PyTorch Lightning"
    )
    parser.add_argument(
        "--dataset_root", default="datasets/chessred", help="Root directory of dataset"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument(
        "--lr_backbone", type=float, default=1e-5, help="Backbone learning rate"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="DETR transformer learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--warmup_iters", type=int, default=1000, help="Warmup iterations"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--output_dir",
        default=f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument("--accelerator", default="auto", help="Accelerator type")
    parser.add_argument("--devices", default="auto", help="Number of devices")
    parser.add_argument("--strategy", default="auto", help="Training strategy")
    parser.add_argument(
        "--visualize_every_n_steps",
        type=int,
        default=150,
        help="Generate validation predictions visualization every N training steps",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
