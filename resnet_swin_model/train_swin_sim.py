# train_swin_sim.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from loader import MedicalSegmentationDataset
from models import SwinUNet_ConvLSTM_SIM
from ddg_virt_user import ddg_virtual_user
from sim import update_sim
from loss_func import DiceCELoss
import torch.nn.functional as F


# Add these near top of file

# ----------------------
# Utility: Class weights
# ----------------------
def compute_class_weights(dataset, num_classes, cache_path="class_weights.npy"):
    if os.path.exists(cache_path):
        print(f"✅ Found cached class weights at {cache_path}")
        weights = np.load(cache_path)
        print("Class weights (cached):", weights)
        return torch.tensor(weights, dtype=torch.float32)

    print("--- Computing class frequencies for weighting ---")
    total_counts = np.zeros(num_classes, dtype=np.float64)
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))

    for i in indices:
        mask = base_dataset[i]["mask"]
        if mask.ndim == 3 and mask.shape[0] == num_classes:
            mask = mask.argmax(dim=0)
        mask = mask.numpy().flatten()
        hist, _ = np.histogram(mask, bins=np.arange(num_classes + 1))
        total_counts += hist

    freq = total_counts / total_counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum()
    np.save(cache_path, weights)
    print("💾 Saved class weights:", weights)
    return torch.tensor(weights, dtype=torch.float32)


# # ----------------------
# # Visualization helper
# # ----------------------
# def inference_and_save(model, dataset, epoch, device, save_dir="outputs"):
#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)
#     idxs = torch.randint(0, len(dataset), (3,))
#     with torch.no_grad():
#         for i, idx in enumerate(idxs):
#             sample = dataset[idx]
#             image = sample["image"].unsqueeze(0).to(device).float()
#             # if image.shape[1] == 1:
#             #     image = image.repeat(1, 3, 1, 1)
#             # scale to [0,1] then normalize to ImageNet
#             image = image / 255.0
#             mean = image.mean(dim=(2, 3), keepdim=True)
#             std = image.std(dim=(2, 3), keepdim=True) + 1e-5
#             image = (image - mean) / std

#             gt_mask = sample["mask"].argmax(dim=0).cpu().numpy()
#             sim_tensor = sample["sim"].unsqueeze(0).to(device).float()
#             logits, _ = model(image, sim_tensor, state=None)
#             pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

#             fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#             axs[0].imshow(gt_mask, cmap="jet")
#             axs[0].set_title("Ground Truth")
#             axs[1].imshow(pred, cmap="jet")
#             axs[1].set_title("Prediction")
#             for ax in axs:
#                 ax.axis("off")
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png"))
#             plt.close()


# ----------------------
# Config
# ----------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")

CLASSES = [f"class_{i}" for i in range(11)]
N_CLASSES = len(CLASSES)
SIM_DEPTH = 4
EPOCHS = 100
BATCH_SIZE = 4
LR = 3e-4
NUM_WORKERS = 2
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# Data
# ----------------------
dataset = MedicalSegmentationDataset(
    pkl_dir=r"/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data",
    classes=CLASSES,
    sim_depth=SIM_DEPTH,
    load_sim=True,
)
val_size = int(len(dataset) * 0.1)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f"Train: {len(train_set)} | Val: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ----------------------
# Model, Loss, Optimizer
# ----------------------
model = SwinUNet_ConvLSTM_SIM(num_classes=N_CLASSES, num_seg_classes=N_CLASSES, sim_depth=SIM_DEPTH, pretrained=True).to(DEVICE)
class_weights = compute_class_weights(train_set, N_CLASSES).to(DEVICE)

criterion = DiceCELoss(num_classes=N_CLASSES, ce_weight=0.4, dice_weight=0.6)
criterion.ce_loss = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

best_val_loss = float("inf")
train_losses, val_losses = [], []
patience, epochs_no_improve = 10, 0

# ----------------------
# Training Loop
# ----------------------
print("--- Starting training loop ---")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(DEVICE)
        gt_masks = batch["mask"].to(DEVICE)
        batch_sim = batch["sim"].to(DEVICE) if batch["sim"] is not None else None

        # image normalization & channel handling
        images = images.float()
        # if images.shape[1] == 1:
        #     images = images.repeat(1, 3, 1, 1)
        # images = images / 255.0
        # images = (images - IMAGENET_MEAN) / IMAGENET_STD
        images = images / 255.0
        mean = images.mean(dim=(2, 3), keepdim=True)
        std = images.std(dim=(2, 3), keepdim=True) + 1e-5
        images = (images - mean) / std


        B = images.size(0)
        num_classes = N_CLASSES
        sim_channels = 2 * SIM_DEPTH * num_classes
        H, W = images.shape[-2], images.shape[-1]

        # prepare sim_batch (B, sim_channels, H, W)
        if batch_sim is not None and batch_sim.dim() == 4:
            sim_batch = batch_sim.clone()
        else:
            sim_batch = torch.zeros((B, sim_channels, H, W), device=DEVICE, dtype=torch.float32)

        # 1) Initial forward to get probs for virtual user
        preds_init, _ = model(images, sim_batch, state=None)
        probs_init = torch.softmax(preds_init, dim=1).detach()

        state = None
        click_prob = 1.0
        for r in range(SIM_DEPTH):       # e.g. 3 simulated user rounds
            logits, state = model(images, sim_batch, state)
            probs = torch.softmax(logits, dim=1).detach()

            for b in range(B):           # for each image in the batch
                interaction_mask, click_prob = ddg_virtual_user(
                    pred_prob=probs[b],
                    gt_mask=gt_masks[b],
                    num_classes=num_classes,
                    click_prob=click_prob,
                    max_clicks=6,
                    drop_rate=0.4
                )
                sim_batch[b] = update_sim(sim_batch[b], interaction_mask, probs[b], num_classes)

        # after all interaction rounds
        final_logits, _ = model(images, sim_batch, state)
        loss = criterion(final_logits, gt_masks.argmax(dim=1).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss /= len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}")


    train_losses.append(train_loss)

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(DEVICE).float()
                gt_masks = batch["mask"].to(DEVICE)
                sim_tensor = batch["sim"].to(DEVICE).float()
            else:
                images, gt_masks = batch
                images = images.to(DEVICE).float()
                gt_masks = gt_masks.to(DEVICE)
                sim_tensor = torch.zeros((images.size(0), 2 * SIM_DEPTH  * N_CLASSES, *images.shape[-2:]), device=DEVICE)
            images = images.float()
            # if images.shape[1] == 1:
            #     images = images.repeat(1, 3, 1, 1)
            # scale to [0,1] then normalize to ImageNet
            # images = images / 255.0
            # images = (images - IMAGENET_MEAN) / IMAGENET_STD
            images = images / 255.0
            mean = images.mean(dim=(2, 3), keepdim=True)
            std = images.std(dim=(2, 3), keepdim=True) + 1e-5
            images = (images - mean) / std


            gt_cls = gt_masks.argmax(dim=1)
            logits, _ = model(images, sim_tensor, state=None)
            val_loss += criterion(logits, gt_cls).item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---------------- Checkpoint ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_path = os.path.join(SAVE_DIR, "best_swin_convlstm_sim_depth_4.pth")
        torch.save(model.state_dict(), best_path)
        print(f"✅ Saved new best model at {best_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"🚨 Early stopping after {patience} epochs without improvement.")
            break

    ckpt_path = os.path.join(SAVE_DIR, f"epoch_depth4_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, ckpt_path)
    print(f"💾 Saved checkpoint at epoch {epoch+1}")

    # inference_and_save(model, dataset, epoch+1, DEVICE, save_dir="epoch_outputs")


# ----------------------
# Plot losses
# ----------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

print("--- Training completed successfully ---")
