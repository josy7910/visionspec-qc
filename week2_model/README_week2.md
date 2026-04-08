# Week 2 — Transfer Learning Model Training

## Goal
Build and train a binary image classifier using Transfer Learning
with ResNet50 to classify magnetic tiles as PASS or DEFECT.

## Files
| File | Purpose |
|---|---|
| `train.py` | Model architecture, training loop, evaluation |

## Model Architecture
Input (224×224×3)
↓
ResNet50 (pretrained on ImageNet, frozen)
↓
GlobalAveragePooling2D
↓
Dense(256, relu)
↓
Dropout(0.5)
↓
Dense(128, relu)
↓
Dropout(0.3)
↓
Dense(1, sigmoid) → 0 = DEFECT, 1 = PASS

## Why Transfer Learning?
Building a CNN from scratch needs millions of images and days of training.
ResNet50 was pre-trained on 1.2 million ImageNet images and already knows
how to detect edges, textures and shapes. We freeze those layers and only
train our custom classification head — this gives high accuracy with
just 863 images in a fraction of the time.

## Training Configuration
| Parameter | Value | Reason |
|---|---|---|
| Base model | ResNet50 | Powerful, well-tested architecture |
| Input size | 224×224 | ResNet50 standard input |
| Batch size | 32 | Balance speed and memory |
| Epochs | 20 (max) | EarlyStopping prevents over-training |
| Optimizer | Adam (lr=0.001) | Adaptive learning rate |
| Loss | Binary Crossentropy | Binary classification |
| Dropout | 0.5 + 0.3 | Prevent overfitting |

## Callbacks
| Callback | Setting | Purpose |
|---|---|---|
| EarlyStopping | patience=5 | Stop if val_loss stagnates |
| ModelCheckpoint | save_best_only | Save best model automatically |
| ReduceLROnPlateau | patience=3, factor=0.5 | Reduce LR if stuck |

## Metrics Tracked
- Accuracy
- Precision
- Recall
- F1 Score (computed from precision + recall)

## How to Run
```bash
# Must run Week 1 first to set up data/
python week2_model/train.py
```

## Expected Output
Base model layers: 175
Trainable layers:  0
Epoch 1/20 — loss: 0.68 — accuracy: 0.61
...
Epoch 15/20 — loss: 0.18 — accuracy: 0.94
MODEL EVALUATION
Loss:      0.1823
Accuracy:  94.21%
Precision: 95.10%
Recall:    93.40%
F1 Score:  94.24%

## Status
✅ Week 2 Complete
