# CheckMateVision - Chess Piece Detection with Deformable DETR

A PyTorch Lightning implementation for training Deformable DETR to detect chess pieces on the ChessReD2k dataset. Early experiments achieve **54.9% mAP** and **74.2% mAP@50** for accurate chess piece detection across 12 piece classes.

## 🏆 Performance Summary

Metrics over test set of 77 images:
- **mAP (0.50:0.95)**: 54.97%
- **mAP@50**: 74.22%
- **mAP@75**: 69.97%
- **Average Recall@100**: 63.91%

These metrics demonstrate strong performance for chess piece detection, with the model effectively distinguishing between all 12 piece types (6 piece types × 2 colors).

## 📊 Key Features

- **Modern Architecture**: Deformable DETR with ResNet-50 backbone from HuggingFace Transformers
- **Efficient Training**: PyTorch Lightning with mixed precision (bf16) and multi-GPU support
- **Comprehensive Monitoring**: TensorBoard integration with loss tracking and prediction visualizations
- **Robust Data Pipeline**: Albumentations-based augmentations with COCO-format annotations
- **Production Ready**: Model checkpointing, validation metrics, and inference utilities

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd CheckMateVision

# Create and activate virtual environment
python -m venv chess_env
source chess_env/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

### 2. Download ChessReD Dataset

The project uses the ChessReD2k dataset which provides bounding box annotations for chess pieces.

```bash
# Download and extract the dataset
python download_chessred.py --output datasets/chessred --extract
```

**Expected dataset structure:**
```
datasets/chessred/
├── annotations.json          # COCO-format annotations
├── images_raw/               # Raw chess board images
│   └── images/
├── chessred_dataset.yaml     # Dataset configuration
└── images.zip               # Compressed images
```

### 3. Training

Start training with the default configuration:

```bash
# Basic training (single GPU)
python train_lightning.py \
    --dataset_root datasets/chessred \
    --epochs 80 \
    --batch_size 2 \
    --output_dir experiments/chess_detection

# Multi-GPU training
python train_lightning.py \
    --dataset_root datasets/chessred \
    --epochs 80 \
    --batch_size 4 \
    --devices 2 \
    --strategy ddp \
    --output_dir experiments/chess_detection_multi_gpu
```

**Training Configuration:**
- **Image Size**: 256×256 (constrained by compute resources)
- **Batch Size**: 2 per GPU (adjustable based on GPU memory)
- **Learning Rate**: 1e-4 (transformer), 1e-5 (backbone)
- **Optimizer**: AdamW with cosine annealing and warmup
- **Augmentations**: Rotation, scaling, color jitter, horizontal flip, Gaussian noise

### 4. Monitoring Training

Training progress is automatically logged to TensorBoard:

```bash
# Launch TensorBoard
tensorboard --logdir experiments/chess_detection

# View in browser at http://localhost:6006
```

**TensorBoard includes:**
- Training and validation loss curves
- Learning rate schedules
- Validation metrics (mAP, mAP50, mAP75, AR)
- Prediction visualizations with bounding boxes
- Model graph and hyperparameters

*[Placeholder for TensorBoard screenshots]*

### 5. Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate_model.py \
    --checkpoint experiments/chess_detection/checkpoints/best.ckpt \
    --dataset_root datasets/chessred \
    --batch_size 4
```

**Evaluation Metrics:**
- COCO-style Average Precision (AP) at multiple IoU thresholds
- Per-class precision and recall
- Confusion matrix for piece type classification
- Detection visualization on test images

## 📁 Project Structure

```
CheckMateVision/
├── train_lightning.py       # Main training script with PyTorch Lightning
├── dataset.py              # ChessPiecesDataset and data utilities
├── validation_utils.py     # COCO evaluation and metrics computation
├── visualization_utils.py  # Prediction visualization tools
├── evaluate_model.py       # Model evaluation script
├── download_chessred.py    # Dataset download utility
├── requirements.txt        # Python dependencies
└── datasets/              # Dataset storage
    └── chessred/         # ChessReD2k dataset
```

## 🎯 Model Architecture

- **Base Model**: Deformable DETR from SenseTime/deformable-detr
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Object Queries**: 32 queries (sufficient for maximum chess pieces on board)
- **Classes**: 12 chess piece types + background
  - White: Pawn, Knight, Bishop, Rook, Queen, King
  - Black: Pawn, Knight, Bishop, Rook, Queen, King
- **Input Resolution**: 256×256 pixels
- **Output Format**: Normalized center coordinates (cx, cy, w, h)

## 📈 Training Details

**Data Augmentation Strategy:**
- Random rotation (±10°)
- Random scaling (±10%)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian noise
- Horizontal flip
- Aspect ratio preserving resize

**Loss Function:**
- Bipartite matching loss (Hungarian algorithm)
- Classification loss (focal loss)
- Bounding box regression loss (L1 + GIoU)

**Optimization:**
- Differential learning rates (backbone vs. transformer)
- Cosine annealing with linear warmup
- Gradient clipping for stability
- Mixed precision training (bf16)

## 🔮 Next Steps

### 1. Digital Chess Board Rendering
- [ ] Implement board state reconstruction from detected pieces
- [ ] Generate digital chess board visualization

### 2. Scale Up Training
- [ ] Train with larger image resolution (512×512 or 800×1333)
- [ ] Utilize cloud computing resources (AWS/GCP/Azure)
- [ ] Experiment with larger batch sizes for improved convergence
- [ ] Multi-node distributed training setup

### 3. Advanced Model Architectures
- [ ] Replace ResNet-50 backbone with DINOv2 for improved feature extraction
- [ ] Experiment with DETR-based variants (RT-DETR, DETA)


## 📊 Detailed Results

```
Dataset: datasets/chessred
Confidence Threshold: 0.05
Test Images: 306
Total Predictions: 9792
Evaluation Time: 11.84 seconds
--------------------------------------------------------------------------------
COCO Metrics:
  mAP (IoU=0.50:0.95): 0.5497
  mAP@50 (IoU=0.50)  : 0.7422
  mAP@75 (IoU=0.75)  : 0.6997
  mAP (small)        : 0.5497
  mAP (medium)       : -1.0000
  mAP (large)        : -1.0000
----------------------------------------
Average Recall:
  AR@1               : 0.4449
  AR@10              : 0.6390
  AR@100             : 0.6391
  AR (small)         : 0.6391
  AR (medium)        : -1.0000
  AR (large)         : -1.0000

```

## 🛠️ Advanced Usage

### Custom Training Configuration

```bash
python train_lightning.py \
    --dataset_root datasets/chessred \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --warmup_iters 1000 \
    --precision bf16-mixed \
    --visualize_every_n_steps 100
```

### Resume Training from Checkpoint

```bash
python train_lightning.py \
    --dataset_root datasets/chessred \
    --resume_from_checkpoint experiments/chess_detection/checkpoints/last.ckpt \
    --epochs 120
```

### Dataset Visualization

```bash
# Visualize dataset samples with annotations
python visualize_dataset.py \
    --dataset_root datasets/chessred \
    --split train \
    --num_samples 10
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- [ChessReD Dataset](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2) for providing high-quality chess piece annotations. References were also taken from the github repository. [Link](https://github.com/tmasouris/end-to-end-chess-recognition/tree/main) 
