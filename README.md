# VisionSpec QC — Automated Visual Quality Control

**Project 3 | Retail — Computer Vision**
**Zaalima Development Pvt. Ltd. | Q4 Production AI Project**

## Overview
VisionSpec QC is a production-grade Computer Vision pipeline that automatically
detects defects in magnetic tiles using Deep Learning. It classifies images as
PASS or DEFECT in real time using a ResNet50-based transfer learning model.

## Project Structure
| Folder | Week | Description |
|---|---|---|
| `week1_data_prep/` | Week 1 | Data augmentation pipeline |
| `week2_model/` | Week 2 | ResNet50 transfer learning model |
| `week3_gradcam/` | Week 3 | GradCAM heatmap visualization |
| `week4_inference/` | Week 4 | Live webcam inference demo |

## Dataset
Magnetic Tile Defect Dataset — 863 images across 5 defect types.

## Results
- Validation accuracy: ~87%
- Batch inference: 10/10 correct
- Live inference: ~7.4 FPS on CPU

## How to Run
```bash
pip install -r requirements.txt
python week1_data_prep/augmentation.py
python week2_model/train.py
python week3_gradcam/gradcam.py
python week4_inference/inference.py
```

## Author
Josy Mol J — Machine Learning Intern, Zaalima Development Pvt. Ltd.