import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os

from loader import MedicalSegmentationDataset
from models import ResNet34_UNet_ConvLSTM

# ----------------------
# Config
# ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = [
    "background",
    "aorta",
    "gall_bladder",
    "kidney_left",
    "kidney_right",
    "liver",
    "pancreas",
    "postcava",
    "spleen",
    "stomach",
]
N_CLASSES = len(CLASSES)
SIM_DEPTH = 3
SAVE_DIR = "results_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# Load dataset
# ----------------------
dataset = MedicalSegmentationDataset(
    pkl_file="/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data",
    classes=CLASSES,
    sim_depth=SIM_DEPTH,
    load_sim=True,
)

# ----------------------
# Load model
# ----------------------
model = ResNet34_UNet_ConvLSTM(
    num_classes=N_CLASSES,
    num_seg_classes=N_CLASSES,
    sim_depth=SIM_DEPTH,
).to(DEVICE)

checkpoint = torch.load("checkpoints/best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# ----------------------
# Pick random samples
# ----------------------
indices = random.sample(range(len(dataset)), 5) #randomly picking 5 slides 

for i, idx in enumerate(indices):
    sample = dataset[idx]

    image = sample["image"].unsqueeze(0).to(DEVICE)   # (1,1,H,W)
    sim   = sample["sim"].unsqueeze(0).to(DEVICE)    # (1,2*D*N,H,W)
    gt_mask = sample["mask"].argmax(dim=0).numpy()   # (H,W)

    # Forward pass
    with torch.no_grad():
        model_input = torch.cat([image, sim], dim=1)   # (1, 1+2*D*N, H, W)
        logits = model(model_input)                   # (1, N, H, W)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # ----------------------
    # Visualization
    # ----------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
    axs[0].set_title("CT Slice")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    # ----------------------
    # Add legend (class-color mapping)
    # ----------------------
    cmap = plt.get_cmap("tab20", N_CLASSES)
    patches = [mpatches.Patch(color=cmap(c), label=CLASSES[c]) for c in range(N_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"sample_{i}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved visualization with legend to {save_path}")
