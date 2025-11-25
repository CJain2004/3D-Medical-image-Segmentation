# 3D Medical Image Segmentation using Sequential Interaction Memory (SIM)

This repository provides a complete pipeline for **interactive 3D medical image segmentation** using automatic deep learning predictions combined with simulated user corrections through **Sequential Interaction Memory (SIM)**.  
The project includes two architectures:

- **ResNet34 Encoder + SIM + ConvLSTM**
- **Swin Transformer Encoder + SIM + ConvLSTM**

The dataset is derived from abdominal CT scans annotated by clinical experts.

---

## 📂 Project Structure

```
3d_medical_img_seg/
│
├── README.md
├── class_distribution.png
├── class_distribution_ignore_bg.png
├── class_weights.npy
├── loss_curve.png
│
├── dataset_distrib.py
├── inference_resnet.py
├── inference_swin_sim.py
│
├── Abdomen_Atlas_Data_prep/
│   ├── compress_files.py
│   ├── dataset.md
│   ├── dataset.py
│   ├── preprocess.py
│
├── resnet_swin_model/
│   ├── advanced_models.py
│   ├── dataset_distrib.py
│   ├── ddg_virt_user.py
│   ├── inference_per_patient.py
│   ├── loader.py
│   ├── loss_func.py
│   ├── plot_loss.py
│   ├── resnet_model.py
│   ├── sim.py
│   ├── swin_model.py
│   ├── train_resnet_sim.py
│   ├── train_swin_sim.py
│
├── sample_visual_results/
│   ├── anim_slice_2181.gif
│   ├── vis_slice_1196.png
│   ├── vis_slice_2092.png
│   ├── vis_slice_2181.png
│   ├── vis_slice_2278.png
│   ├── vis_slice_743.png
│   ├── vis_slice_805.png
│
└── TMH_dataset_prep/
    ├── data_prep.py
    ├── preprocess.py
```

---

## 🧠 Overview

Medical image segmentation is essential for diagnosis, surgery planning, and monitoring.  
Manual annotation is slow and expensive, so this project focuses on **interactive segmentation**, where:

- The model generates an automatic segmentation.
- A **virtual user** detects mistakes and adds corrective clicks.
- These correction maps + logits accumulate as a **Sequential Interaction Memory (SIM)**.
- A **ConvLSTM decoder** refines predictions for multiple rounds.

This workflow mimics how clinicians iteratively correct segmentation outputs.

---

## 🚀 Key Features

### ✔ Sequential Interaction Memory (SIM)
- Stores ordered **click maps** and **logit maps**
- Preserves the sequence of corrections
- Injected into the decoder at each resolution level
- Enables multi-round refinement

### ✔ Dynamic Data Generation (DDG)
Virtual user simulation includes:
- Detecting error regions
- Centroid-based clicking
- Probabilistic skipping
- Click dropping (80%) for realistic noise
- Up to `max_clicks` per class

### ✔ Architectures
- **ResNet34-inspired encoder**
- **Swin Transformer encoder** (via TIMM pretrained backbone)
- Shared ConvLSTM + U-Net–like decoder

### ✔ Dataset Pipeline
Includes:
- Hounsfield unit clipping  
- Resampling  
- Normalization  
- Slice extraction  
- Label remapping  
- Class distribution computation  
- Weight generation for loss functions  

---

## 🧪 Training

### Train ResNet + SIM
```bash
python resnet_swin_model/train_resnet_sim.py
```

### Train Swin + SIM
```bash
python resnet_swin_model/train_swin_sim.py
```

### Training Details
- **Optimizer:** AdamW  
- **Loss (ResNet):** Dice + Tversky  
- **Loss (Swin):** Dice + Weighted Cross Entropy  
- **Learning rate:** 3e-4  
- **Scheduler:** ReduceLROnPlateau  
- **Batch size:** 4  
- **Interaction rounds:** 3–4  
- **Early stopping:** patience = 10  

---

## 🔍 Inference

### ResNet inference
```bash
python inference_resnet.py
```

### Swin inference
```bash
python inference_swin_sim.py
```

### Per-patient inference
```bash
python resnet_swin_model/inference_per_patient.py
```

Outputs include:
- Segmentation masks  
- Overlay visualizations  
- GIF animations  
- Per-class Dice scores  

---

## 📊 Results Summary

- **ResNet + SIM** achieved strong baseline predictions and consistent improvements across interaction rounds.
- **Swin + SIM** improved but its initial predictions were weaker, limiting gains.
- Severe class imbalance was handled using **class-weighted CE** and **Tversky-based losses**.

Visual examples are available inside `sample_visual_results/`.

---

## 📈 Provided Visualizations

- `class_distribution.png`  
- `class_distribution_ignore_bg.png`  
- `loss_curve.png`  
- GIF and slice outputs in `sample_visual_results/`  

---

## 🔮 Future Work

- Slice-consistent attention across patients  
- Balanced dataset for smaller organ classes  
- SAM2-style prompt embedding for SIM  
- Deformable attention in the decoder  
- Improved anatomical continuity modeling  

---

## 👤 Author

**Cherish Jain (22B3937)**  
IIT Bombay — Electrical Engineering  
Email: **cherishjain01@gmail.com**
