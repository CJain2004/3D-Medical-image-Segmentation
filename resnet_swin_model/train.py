import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from loader import MedicalSegmentationDataset
from models import ResNet34_UNet_ConvLSTM
from ddg_virt_user import ddg_virtual_user
from sim import update_sim
from loss_func import DiceCELoss
import matplotlib.pyplot as plt

def inference_and_save(model, dataset, epoch, device, save_dir="outputs"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # take a few random samples
    idxs = torch.randint(0, len(dataset), (3,))   # 3 samples per epoch
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device).float()
            image = image / 255.0
            image = (image - 0.5) / 0.5
            gt_mask = sample["mask"].argmax(dim=0).cpu().numpy()  # (H, W)

            sim_tensor = sample["sim"].unsqueeze(0).to(device).float()
            model_input = torch.cat([image, sim_tensor], dim=1)
            logits = model(model_input)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

            # plot side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(gt_mask, cmap="jet")
            axs[0].set_title("Ground Truth")
            axs[1].imshow(pred, cmap="jet")
            axs[1].set_title("Prediction")
            for ax in axs: ax.axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png"))
            plt.close()

print("--- Script starting ---")

# ----------------------
# Config
# ----------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")
torch.backends.cudnn.benchmark = True

CLASSES = [f"class_{i}" for i in range(11)]

N_CLASSES = len(CLASSES)
SIM_DEPTH = 3
EPOCHS = 75
BATCH_SIZE = 4
LR = 1e-4
NUM_WORKERS = 2

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# Data
# ----------------------
print("--- Initializing dataset ---")

train_dataset = MedicalSegmentationDataset(
    pkl_dir=r"/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data",  # folder containing many .pkl
    classes=CLASSES,
    sim_depth=SIM_DEPTH,
    load_sim=True,
)
print(f"--- Dataset initialized. Found {len(train_dataset)} samples. ---")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=2 ,         # add this
    persistent_workers=True,
    pin_memory=True if DEVICE.type == "cuda" else False, 
)
# ----------------------
# Model, Loss, Optimizer
# ----------------------
# print("--- Creating model ---")

model = ResNet34_UNet_ConvLSTM(
    num_classes=N_CLASSES,
    num_seg_classes=N_CLASSES,
    sim_depth=SIM_DEPTH,
    use_deconv=True,
)




model = model.to(DEVICE)
print("--- Model created and moved to device ---")

criterion = DiceCELoss(num_classes=N_CLASSES, ce_weight=0.4, dice_weight=0.6)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# ----------------------
# Scheduler & Early stopping config
# ----------------------
patience = 10      # stop if no improvement for 10 epochs
best_loss = float("inf")
epochs_no_improve = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ----------------------
# Training Loop
# ----------------------
best_loss = float("inf")
print("--- Starting training loop ---")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(DEVICE, non_blocking=True).float()
        images = images / 255.0
        images = (images - 0.5) / 0.5
        gt_masks_oh = batch["mask"].to(DEVICE, non_blocking=True)  # (B, N, H, W)
        sim_tensor = batch["sim"].to(DEVICE, non_blocking=True).float() # (B, 2*D*N, H, W)

        gt_masks = gt_masks_oh.argmax(dim=1)  # (B, H, W)

        B = images.size(0)
        click_prob = [1.0 for _ in range(B)]
        logits = None

        for round_idx in range(SIM_DEPTH):
            model_input = torch.cat([images, sim_tensor], dim=1)
            logits = model(model_input)             # (B, N, H, W)
            probs = torch.softmax(logits, dim=1)

            if round_idx < SIM_DEPTH - 1:
                for b in range(B):
                    inter_mask_b, new_click_prob = ddg_virtual_user(
                        pred_prob=probs[b],
                        gt_mask=gt_masks_oh[b],
                        num_classes=N_CLASSES,
                        click_prob=click_prob[b],
                    )
                    click_prob[b] = new_click_prob
                    sim_tensor[b] = update_sim(sim_tensor[b], inter_mask_b, probs[b], N_CLASSES)

        # ----- Loss -----
        loss = criterion(logits, gt_masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

    # Scheduler step
    scheduler.step(epoch_loss)
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")


    # --- Checkpointing ---
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0
        best_path = os.path.join(SAVE_DIR, "best_gpu_0_resnet.pth")
        torch.save(model.state_dict(), best_path)
        print(f"✅ Saved new best model at {best_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"🚨 Early stopping triggered after {patience} epochs without improvement.")
            break

    ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
    }, ckpt_path)
    print(f"💾 Saved checkpoint at epoch {epoch+1}")

    inference_and_save(model, train_dataset, epoch+1, DEVICE, save_dir="epoch_outputs")

print("--- Training completed successfully ---")

