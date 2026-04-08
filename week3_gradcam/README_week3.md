# Week 3 — GradCAM Heatmap Visualization

## Goal
Implement visual explainability for the trained model using Class Activation
Mapping (CAM) to show WHERE the model looks when making pass/defect decisions.
This builds trust — engineers can verify the model focuses on actual defect
regions, not random background noise.

## Files
| File | Purpose |
|---|---|
| `gradcam.py` | Full GradCAM pipeline — load, predict, visualize, save |

## What is GradCAM?
GradCAM (Gradient-weighted Class Activation Mapping) answers the question:
"Which part of the image did the AI focus on when it made this decision?"

It works by:
1. Passing the image through the model
2. Looking at the last convolutional layer's feature maps
3. Measuring which spatial regions had the highest activation
4. Drawing a colour heatmap over those regions

## Colour Meaning
| Colour | Meaning |
|---|---|
| Red / Orange | Model focused here most |
| Yellow / Green | Model paid some attention here |
| Blue | Model ignored this area |

## Output — 3 Panel Image
Each GradCAM output shows 3 panels side by side:

**Panel 1 — Original Image:** The raw magnetic tile photo as fed into the model.

**Panel 2 — Activation Heatmap:** A 7×7 grid showing which spatial regions
had the most activation. ResNet50's last conv layer produces 7×7 feature maps
which get upscaled to 224×224 for display.

**Panel 3 — Overlay:** The heatmap blended on top of the original image so
you can see exactly which part of the tile the model was analysing.

## Key Observation From Results
PASS tiles and DEFECT tiles produce clearly opposite heatmap patterns:

| Class | Heatmap Pattern | Meaning |
|---|---|---|
| PASS | Red in centre, blue on edges | Model focused on smooth central texture |
| DEFECT | Red on edges, blue in centre | Model detected abnormal boundary regions |

This inverted pattern confirms the model learned meaningful differences
between the two classes — not just random noise.

## Why ResNet50 Base is Frozen
ResNet50 was pre-trained on 1.2 million ImageNet images and already knows
how to detect edges, textures and shapes. Freezing its layers preserves this
knowledge. If we unfroze all layers and trained on only 863 tile images, the
model would forget its pre-trained knowledge and perform worse. We only train
the custom classification head on top.

## How to Run
```bash
# Must have models/visionspec_model.h5 from Week 2
python week3_gradcam/gradcam.py
```

## Expected Output
Loading model from models/visionspec_model.h5...
Model loaded ✅
Running GradCAM on sample images...
[1/6]
Image     : pass_0000.jpg
Prediction: PASS ✅ (72.9% confidence)
Saved: week3_gradcam/outputs/gradcam_1.png
[2/6]
Image     : pass_0001.png
Prediction: PASS ✅ (82.3% confidence)
Saved: week3_gradcam/outputs/gradcam_2.png
...
Week 3 complete! ✅
Heatmaps saved in week3_gradcam/outputs/

## Output Files
| File | Description |
|---|---|
| `week3_gradcam/outputs/gradcam_1.png` | Heatmap for image 1 |
| `week3_gradcam/outputs/gradcam_2.png` | Heatmap for image 2 |
| `week3_gradcam/outputs/gradcam_3.png` | Heatmap for image 3 |
| `week3_gradcam/outputs/gradcam_4.png` | Heatmap for image 4 |
| `week3_gradcam/outputs/gradcam_5.png` | Heatmap for image 5 |
| `week3_gradcam/outputs/gradcam_6.png` | Heatmap for image 6 |

## Status
✅ Week 3 Complete