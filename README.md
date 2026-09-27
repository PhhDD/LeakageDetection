# Gate Leakage Detection in Quantum-Dot Charge Stability Diagrams

A computer vision project for detecting gate leakage in quantum-dot charge
stability diagrams using a fine-tuned ResNet18 classifier.

## Overview

Gate leakage can make a quantum-dot device unsuitable for further
measurements. Manually screening large numbers of charge stability diagrams
is time-consuming, so this project investigates whether a CNN can
automatically identify measurements exhibiting gate leakage.

The input to the model is a cropped grayscale charge stability diagram.
The model predicts the probability that the measurement contains gate
leakage.

## Problem

A charge stability diagram can be represented as a 2D current map:

- **x-axis:** gate voltage
- **y-axis:** source-drain voltage
- **pixel intensity:** measured current

The characteristic feature of gate leakage is a dark-current region that
persists vertically through a substantial fraction of the diagram.

The task is therefore formulated as binary image classification:

| Label | Meaning |
|---|---|
| `0` | No gate leakage |
| `1` | Gate leakage |

## Dataset

The labelled dataset contains **375 charge stability diagrams**:

- 295 without leakage
- 80 with leakage
- 21.3% positive examples

The data were split into training, validation and test sets using a
stratified 70/15/15 split with a fixed random seed.

The final evaluation set contains 57 images:

- 45 no leakage
- 12 leakage

## Preprocessing

The original plots contain axes, labels and other plotting elements that
are not relevant to the classification task.

The relevant heatmap region is therefore cropped before being passed to the
model.

The preprocessing pipeline is:

1. Crop the charge stability diagram.
2. Convert to grayscale.
3. Resize to `224 × 224`.
4. Replicate the grayscale image across three channels.
5. Apply the input normalization used by the ResNet18 model.

Keeping the image grayscale focuses the model on the spatial structure of
the current map rather than the absolute colour representation of the plot.

## Model

The classifier is based on **ResNet18 pretrained on ImageNet**.

Rather than training the complete network from scratch, transfer learning
was used:

- ResNet18 convolutional backbone
- Replace the original 1000-class classification head with a single-output
  binary classifier
- Initially freeze the backbone
- Fine-tune `layer4` and the final fully connected layer

The final training setup used:

- Optimizer: AdamW
- `layer4` learning rate: `1e-4`
- Classification head learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 16
- CPU-compatible PyTorch implementation

### Data augmentation

Vertical flipping was used as the main augmentation.

The original training images were retained and vertically flipped versions
were added to the training set.

Validation and test images were **not augmented**.

## Evaluation

The model was evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 score
- Confusion matrix

The original fine-tuned model achieved:

| Metric | Test |
|---|---:|
| ROC-AUC | **0.983** |
| PR-AUC | **0.964** |
| Precision | **0.917** |
| Recall | **0.917** |
| F1 | **0.917** |

The test set contained only 12 positive examples, so these metrics should be
interpreted with that sample size in mind.

## Decision Threshold

In the intended experimental workflow, false positives are particularly
costly.

A false positive can cause a potentially usable device to be discarded,
whereas a false negative means that an additional device may need to be
checked manually.

The operating threshold was therefore selected on the validation set with
an emphasis on controlling false positives.

The final threshold is:

```text
P(leakage) >= 0.67  →  potential leakage
P(leakage) <  0.67  →  keep