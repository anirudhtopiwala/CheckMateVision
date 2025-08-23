# CheckMateVision - Chess Piece Detection with Deformable DETR

A PyTorch Lightning implementation for training Deformable DETR to detect chess pieces on the ChessReD2k dataset. Early experiments achieve **54.9% mAP** and **74.2% mAP@50** for accurate chess piece detection across 12 piece classes.

## Motivation 
Being able to render a digital chess board from an image of a physical chess board has multiple applications in online chess streaming, game analysis, and augmented reality. This project is part of a bigger project to have a a robotic arm play chess against a human opponent. The first step is to accurately detect the chess pieces on the board.

Existing approaches for chess piece detection often rely on traditional computer vision techniques or simpler deep learning models. [Chesscog](https://github.com/georg-wolflein/chesscog?tab=readme-ov-file) for example uses a combination of image processing and a CNN classifier to identify pieces. These fail to generalize across different views. [ChessReD](https://github.com/tmasouris/end-to-end-chess-recognition/tree/main) proposes an end-to-end pipeline using ResNeXt backbone and and provides a high quality dataset with bounding box annotations for chess pieces. The paper indicates not having success with DETR because of small bounding box predictions, and therefore this project aims to explore the performance of Deformable DETR on this task. 


## 🏆 Performance Summary
Metrics over test set of 306 images or chess boards:
- **mAP (0.50:0.95)**: 54.97%
- **mAP@50**: 74.22%
- **mAP@75**: 69.97%
- **Average Recall@100**: 63.91%
- **Boards with 0 mistakes**: 8.2%
- **Boards with ≤1 mistake**: 25.8%
- **Average mistakes per board**: 4.25

These metrics demonstrate strong performance for chess piece detection, with the model effectively distinguishing between all 12 piece types (6 piece types × 2 colors).

The below table compares the performance of this Deformable DETR model with Chesscog and ChessReD's model. 

(*The paper aggregates the metrics across the entire test set. However, the Deformable DETR metrics are calculated for the test set under ChessReD2k which has bbox annotations for all pieces.*)

| Metric | Chesscog | ResNeXt | Deformable DETR |
|--------|----------|---------|------------------|
| Mean incorrect squares per board | 42.87 | 3.40 | 4.25 |
| Boards with no mistakes (%) | 2.30% | 15.26% | 8.20% | 
| Boards with ≤ 1 mistake (%) | 7.79% | 25.92% | 25.80% |

Examples of model predictions on test images:
![Model Predictions](assets/eval_test_set.gif)

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
    --image_size 256 \
    --output_dir experiments/chess_detection

# Multi-GPU training with larger image size
python train_lightning.py \
    --dataset_root datasets/chessred \
    --epochs 80 \
    --batch_size 4 \
    --image_size 512 \
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
tensorboard --logdir experiments/chess_detection
```

**TensorBoard includes:**
- Training and validation loss curves
- Learning rate schedules
- Validation metrics (mAP, mAP50, mAP75, AR)
- Prediction visualizations with bounding boxes
- Model graph and hyperparameters

### 5. Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate_model.py \
    --checkpoint experiments/chess_detection/checkpoints/best.ckpt \
    --dataset_root datasets/chessred \
    --batch_size 4
```

Metrics include mAP, mAP@50, mAP@75, Average Recall, and board-level mistake analysis.


## 🎯 Model Architecture

- **Base Model**: Deformable DETR from SenseTime/deformable-detr
- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Object Queries**: 32 queries (sufficient for maximum chess pieces on board)
- **Classes**: 12 chess piece types + background
  - White: Pawn, Knight, Bishop, Rook, Queen, King
  - Black: Pawn, Knight, Bishop, Rook, Queen, King
- **Input Resolution**: 256×256 pixels
- **Output Format**: Normalized center coordinates (cx, cy, w, h)

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
EVALUATION RESULTS
================================================================================
Checkpoint: CheckMateVision/experiments/20250816_235942/checkpoints/best-epoch=89-val_mAP50=0.740.ckpt
Dataset: datasets/chessred
Confidence Threshold: 0.3
Test Images: 306
Total Predictions: 8212
Evaluation Time: 11.76 seconds
--------------------------------------------------------------------------------
COCO Metrics:
  mAP (IoU=0.50:0.95): 0.5422
  mAP@50 (IoU=0.50)  : 0.7319
  mAP@75 (IoU=0.75)  : 0.6908
  mAP (small)        : 0.5422
  mAP (medium)       : -1.0000
  mAP (large)        : -1.0000
----------------------------------------
Average Recall:
  AR@1               : 0.4396
  AR@10              : 0.6256
  AR@100             : 0.6257
  AR (small)         : 0.6257
  AR (medium)        : -1.0000
  AR (large)         : -1.0000
----------------------------------------
Board-level Metrics:
  Boards with 0 mistakes      : 8.2%
  Boards with ≤1 mistake      : 25.8%
  Average mistakes per board   : 4.25
  Total boards analyzed        : 306
  Total mistakes detected      : 1299
================================================================================
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
