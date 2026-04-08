# Week 1 — Data Preparation & Augmentation

## Goal
Set up the image dataset pipeline with real-time data augmentation
to ensure the model generalises across lighting and camera variability.


## Files
| File | Purpose |
|---|---|
| `augmentation.py` | Data augmentation pipeline + dataset loader |
| `create_dummy_data.py` | Generates placeholder images for testing |


## Dataset
**Magnetic Tile Defect Dataset** (surface-defect-detection-dataset)
- Source: Kaggle
- Total images: 863
- Classes: pass (285) and defect (578 across 4 defect types)

| Folder | Label | Count |
|---|---|---|
| MT_Free | pass | 285 |
| MT_Blowhole | defect | 230 |
| MT_Break | defect | 170 |
| MT_Crack | defect | 114 |
| MT_Fray | defect | 64 |


## Data Split
data/
├── train/
│   ├── pass/     → 228 images (80%)
│   └── defect/   → 462 images (80%)
└── val/
├── pass/     →  57 images (20%)
└── defect/   → 116 images (20%)


## Augmentation Applied (training only)
| Technique | Value | Reason |
|---|---|---|
| Rescale | 1/255 | Normalize pixels 0–1 |
| Rotation | ±20° | Camera angle variation |
| Zoom | ±15% | Distance variation |
| Width shift | ±10% | Horizontal camera offset |
| Height shift | ±10% | Vertical camera offset |
| Horizontal flip | True | Mirror symmetry |
| Brightness | 0.8–1.2 | Lighting variation |
| Fill mode | nearest | Edge pixel filling |

Validation data has **no augmentation** — only rescaling for honest evaluation.


## How to Run
```bash
# Generate dummy data for testing (optional)
python week1_data_prep/create_dummy_data.py

# Run augmentation pipeline and visualise samples
python week1_data_prep/augmentation.py
```

## Expected Output
Found 690 images belonging to 2 classes.
Found 173 images belonging to 2 classes.
Class mapping: {'defect': 0, 'pass': 1}
Training images:   690
Validation images: 173
Train set: 690 images
defect →  462 images (67.0%)
pass   →  228 images (33.0%)
Val set: 173 images
defect →  116 images (67.1%)
pass   →   57 images (32.9%)

Sample augmented images saved to `week1_data_prep/augmented_samples.png`


## Status
✅ Week 1 Complete
