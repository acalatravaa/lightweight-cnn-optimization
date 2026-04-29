# Lightweight CNN Optimization for Classification and Detection

This repository contains the architectural analysis and optimization of lightweight Convolutional Neural Networks (CNNs) designed for edge-computing environments. The project is divided into two distinct stages: modifying a MobileNetV1 backbone for image classification, and optimizing a YOLO-based detector head for object detection.

---

## Stage 1: Backbone Optimization (CIFAR-100)
The primary objective of the first stage was to optimize a MobileNetV1 architecture on the CIFAR-100 dataset while strictly adhering to a high initial learning rate constraint of 0.1.

### Architecture Modifications
The baseline MobileNetV1 model suffered from "manifold collapse" in low-dimensional spaces, yielding a baseline top-1 accuracy of 67.09%. To surpass the 70% accuracy threshold, the following structural and regularization upgrades were implemented:
* **Linear Bottlenecks:** Removed the non-linear ReLU activation from pointwise convolutions to preserve feature information.
* **Capacity Expansion:** Applied a width multiplier ($\alpha=2.0$) to double the channel capacity across all layers.
* **Regularization Synergy:** Integrated PyTorch's CIFAR-10 AutoAugment policy, a Dropout layer (p=0.2), and Label Smoothing (0.1) to prevent the widened network from overfitting.
* **Delayed Learning Schedule:** Shifted learning rate decay milestones to allow the network sufficient time to map the heavily augmented dataset.

<img width="700" height="185" alt="image" src="https://github.com/user-attachments/assets/d96d9f65-3a38-4402-8708-a02a293c3c83" />
<img width="700" height="181" alt="image" src="https://github.com/user-attachments/assets/f0d15967-223c-4ae2-ba8e-6a330e1f38d5" />


### Performance Results
The final optimized architecture (`WideLinearMobileNet`) achieved a peak top-1 accuracy of **73.95%**.

<img width="700" height="149" alt="image" src="https://github.com/user-attachments/assets/3461f812-1bc1-402b-83e3-b886db797b4d" />


| Architecture | Modifications | Peak Accuracy |
| :--- | :--- | :--- |
| Baseline | Standard MobileNetV1 | 67.09% |
| `base_linear` | Linear Bottlenecks Only | 67.97% |
| `wide_linear` | Linear Bottlenecks, $\alpha=2.0$, AutoAugment, Label Smoothing | **73.95%** |



### Execution & Reproducibility (Stage 1)
Ensure you navigate into the Stage 1 directory so the local modules are properly recognized:
```bash
cd stage1_classification
```
**1. Training the Model:** Train the optimized `wide_linear` model from scratch.
```bash
python train.py -net wide_linear -b 128 -lr 0.1 -gpu
```
**2. Optimal Learning Rate Search:** Reproduce the learning rate landscape visualization.
```bash
python lr_finder.py -net wide_linear -b 64 -max_lr 10.0 -gpu
```
**3. Evaluation:** Generate Top-1 and Top-5 accuracy metrics (point to your generated `.pth` weights).
```bash
python test.py -net wide_linear -weights checkpoint/wide_linear/YOUR_TIMESTAMP/100-regular.pth -gpu
```

---

## Stage 2: Detector Module Optimization (PASCAL VOC07)
The second phase transitions to object detection. The feature-extracting backbone was strictly fixed as a pre-trained MobileNetV2, shifting the engineering focus entirely to optimizing the YOLO-based detector module to maximize Mean Average Precision (mAP) without losing real-time inference capabilities.

### Architectural Bottlenecks & Solutions
The baseline model (Coupled Head + Element-wise Addition + MSE Loss) yielded a 0.529 mAP and suffered from a high false-positive proposal rate. To resolve these mathematical and structural bottlenecks, three specific upgrades were validated via an ablation study:
* **FPN Concatenation:** Replaced element-wise addition with channel-wise concatenation to prevent deep semantic features from destructively interfering with shallow spatial features.
* **Geometrically Aligned Loss (CIoU):** Replaced Mean Squared Error (MSE) with Complete Intersection over Union (CIoU) loss. This provided a non-vanishing gradient that penalized bounding box overlap discrepancies, center-point distances, and aspect ratio mismatches.
* **Decoupled Detection Head:** Resolved the inherent mathematical conflict between classification (translation invariance) and localization (translation variance) by physically separating the detection head into two parallel convolutional branches.

### Performance Results
The finalized architecture synergized all three modifications, achieving a robust mAP of **0.575** (+4.6% absolute gain). Furthermore, the total predicted bounding box count dropped from ~14,500 to 11,708, demonstrating the network learned to regress targets with significantly higher confidence and precision.



| Model Architecture | mAP (IoU=0.5) | Gain vs. Baseline | Inference | Predicted Boxes |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Coupled + MSE) | 0.529 | - | **65 FPS** | ~14,500 |
| Final Combined Model | **0.575** | **+4.6%** | 40 FPS | **~11,708** |

*Note: While the Decoupled Head increased the parameter count and dropped inference speed from 65 FPS to 40 FPS, the model remains comfortably above the 30 FPS threshold for real-time video deployment, making the mAP gain a highly favorable edge-computing trade-off.*

### Execution & Reproducibility (Stage 2)
Navigate into the Stage 2 directory before executing the scripts:
```bash
cd stage2_detection
```
**1. Data Preparation:** Convert the PASCAL VOC dataset into the optimized LMDB format.
```bash
python folder2lmdb.py
```
**2. Training:** Train the final optimized object detector.
```bash
python train.py
```
**3. Inference:** Run inference and visualize bounding box predictions on test images.
```bash
python inference.py -c checkpoint/model_best.pth.tar -i path/to/test/image.jpg
```
```
