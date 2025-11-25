# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import random
# import os
# import glob
# import shutil

# from loader import MedicalSegmentationDataset
# from models import ResNet34_UNet_ConvLSTM
# from ddg_virt_user import ddg_virtual_user
# from sim import update_sim

# # ----------------------
# # Config
# # ----------------------
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# CLASSES = [f"class_{i}" for i in range(11)]

# N_CLASSES = len(CLASSES)
# SIM_DEPTH = 3
# SAVE_DIR = "results_vis"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # 🔹 Clear existing files in SAVE_DIR
# for f in os.listdir(SAVE_DIR):
#     file_path = os.path.join(SAVE_DIR, f)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print(f"⚠️ Failed to delete {file_path}. Reason: {e}")

# # ----------------------
# # Pick 2-3 random patient pickle files
# # ----------------------
# PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data"
# all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))

# if len(all_pkls) == 0:
#     raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

# chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))
# chosen_names = [os.path.basename(f) for f in chosen_pkls]
# print(f"✅ Selected patient files: {chosen_names}")

# # ----------------------
# # Load dataset
# # ----------------------
# dataset = MedicalSegmentationDataset(
#     pkl_dir=PKL_DIR,
#     classes=list(range(N_CLASSES)),
#     sim_depth=SIM_DEPTH,
#     load_sim=True,
# )

# # Filter dataset indices belonging only to chosen patients
# chosen_indices = [
#     i for i, (fpath, pid, slice_idx) in enumerate(dataset.slice_index)
#     if os.path.basename(fpath) in chosen_names
# ]

# # ----------------------
# # Load model
# # ----------------------
# model = ResNet34_UNet_ConvLSTM(
#     num_classes=N_CLASSES,
#     num_seg_classes=N_CLASSES,
#     sim_depth=SIM_DEPTH,
# ).to(DEVICE)

# checkpoint = torch.load("checkpoints/epoch_2.pth", map_location=DEVICE)
# if "model_state_dict" in checkpoint:
#     model.load_state_dict(checkpoint["model_state_dict"])
# else:
#     model.load_state_dict(checkpoint)
# model.eval()

# # ----------------------
# # Loop over chosen patients
# # ----------------------
# for pkl_name in chosen_names:
#     print(f"\n📂 Processing patient file: {pkl_name}")

#     # find all slice indices for this patient
#     patient_indices = [
#         i for i, (fpath, pid, slice_idx) in enumerate(dataset.slice_index)
#         if os.path.basename(fpath) == pkl_name
#     ]
#     indices = random.sample(patient_indices, min(5, len(patient_indices)))

#     for i, idx in enumerate(indices):
#         sample = dataset[idx]

#         image = sample["image"].unsqueeze(0).to(DEVICE)   # (1,1,H,W)
#         sim   = sample["sim"].unsqueeze(0).to(DEVICE)    # (1,2*D*N,H,W)
#         gt_mask = sample["mask"].argmax(dim=0).numpy()   # (H,W)

#         click_prob = 1.0
#         round_preds = []
#         round_clicks = []

#         # ----------------------
#         # SIM + Virtual User Loop
#         # ----------------------
#         for round_idx in range(SIM_DEPTH):
#             with torch.no_grad():
#                 model_input = torch.cat([image, sim], dim=1)
#                 logits = model(model_input)
#                 pred_prob = torch.softmax(logits[0], dim=0)
#                 pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()


#             # Virtual user interaction
#             gt_onehot = sample["mask"].to(DEVICE)
#             interaction_mask, click_prob = ddg_virtual_user(
#                 pred_prob.cpu(), gt_onehot.cpu(), N_CLASSES, click_prob=click_prob
#             )

#             click_coords = np.argwhere(interaction_mask.numpy() == 1)
#             round_clicks.append(click_coords)
#             round_preds.append(pred)

#             # Update SIM
#             sim[0] = update_sim(sim[0], interaction_mask.to(DEVICE), pred_prob.to(DEVICE), N_CLASSES)

#         # Final prediction after all rounds
#         with torch.no_grad():
#             final_input = torch.cat([image, sim], dim=1)
#             final_logits = model(final_input)
#             final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

#         # ----------------------
#         # Visualization
#         # ----------------------
#         ncols = 3 + SIM_DEPTH + 1  # CT + GT + Initial + rounds + Final
#         fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 5))

#         # Original CT
#         axs[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
#         axs[0].set_title("CT Slice")
#         axs[0].axis("off")

#         # Ground Truth
#         axs[1].imshow(gt_mask, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[1].set_title("Ground Truth")
#         axs[1].axis("off")

#         # Initial Prediction
#         axs[2].imshow(round_preds[0], cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[2].set_title("Initial Prediction")
#         axs[2].axis("off")

#         # Predictions after each round
#         for r in range(SIM_DEPTH):
#             axs[3+r].imshow(round_preds[r], cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#             axs[3+r].set_title(f"Round {r+1}")
#             axs[3+r].axis("off")

#             # Mark clicks
#             coords = round_clicks[r]
#             for c, y, x in coords:
#                 axs[3+r].plot(x, y, "rx", markersize=6, markeredgewidth=2)

#         # Final Prediction
#         axs[-1].imshow(final_pred, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[-1].set_title("Final Prediction")
#         axs[-1].axis("off")

#         # Legend
#         cmap = plt.get_cmap("tab20", N_CLASSES)
#         patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}") for c in range(N_CLASSES)]
#         fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

#         plt.tight_layout()
#         save_path = os.path.join(SAVE_DIR, f"{pkl_name}_slice_{i}_all.png")
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()
#         print(f"   ✅ Saved complete visualization: {save_path}")



# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import random
# import os
# import glob
# import shutil

# from loader import MedicalSegmentationDataset
# from models import ResNet34_UNet_ConvLSTM
# from ddg_virt_user import ddg_virtual_user
# from sim import update_sim

# # ----------------------
# # Config
# # ----------------------
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# CLASSES = [f"class_{i}" for i in range(11)]
# N_CLASSES = len(CLASSES)
# SIM_DEPTH = 3
# SAVE_DIR = "results_vis"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # 🔹 Clear existing files in SAVE_DIR
# for f in os.listdir(SAVE_DIR):
#     file_path = os.path.join(SAVE_DIR, f)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print(f"⚠️ Failed to delete {file_path}. Reason: {e}")

# # ----------------------
# # Pick 2-3 random patient pickle files
# # ----------------------
# PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_test_data"
# all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))

# if len(all_pkls) == 0:
#     raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

# chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))
# chosen_names = [os.path.basename(f) for f in all_pkls]
# print(f"✅ Selected patient files: {chosen_names}")

# # ----------------------
# # Load dataset
# # ----------------------
# dataset = MedicalSegmentationDataset(
#     pkl_dir=PKL_DIR,
#     classes=list(range(N_CLASSES)),
#     sim_depth=SIM_DEPTH,
#     load_sim=True,
# )

# # ----------------------
# # Load model
# # ----------------------
# model = ResNet34_UNet_ConvLSTM(
#     num_classes=N_CLASSES,
#     num_seg_classes=N_CLASSES,
#     sim_depth=SIM_DEPTH,
# ).to(DEVICE)

# checkpoint = torch.load("checkpoints/best_gpu_0.pth", map_location=DEVICE)
# if "model_state_dict" in checkpoint:
#     model.load_state_dict(checkpoint["model_state_dict"])
# else:
#     model.load_state_dict(checkpoint)
# model.eval()

# # ----------------------
# # Dice function
# # ----------------------
# def dice_score_per_class(pred, gt, num_classes):
#     dice_per_class = np.zeros(num_classes)
#     for c in range(num_classes):
#         pred_c = (pred == c).astype(np.uint8)
#         gt_c = (gt == c).astype(np.uint8)
#         intersection = (pred_c * gt_c).sum()
#         union = pred_c.sum() + gt_c.sum()
#         dice = (2 * intersection / union) if union > 0 else np.nan
#         dice_per_class[c] = dice
#     return dice_per_class


# # ----------------------
# # Randomly pick 30–40 slices across chosen patients
# # ----------------------
# all_indices = []
# for pkl_name in chosen_names:
#     patient_indices = [
#         i for i, (fpath, pid, slice_idx) in enumerate(dataset.slice_index)
#         if os.path.basename(fpath) == pkl_name
#     ]
#     all_indices.extend(patient_indices)

# num_slices =  min(30,len(all_indices))
# # num_slices =  len(all_indices)
# chosen_indices = random.sample(all_indices, num_slices)
# print(f"\n🎯 Selected {len(chosen_indices)} total slices for evaluation\n")

# # ----------------------
# # Loop: run inference, compute Dice, and visualize a few slices
# # ----------------------
# all_dice_scores = []
# visualization_indices = random.sample(chosen_indices, min(6, len(chosen_indices)))  # visualize only few

# for idx in chosen_indices:
#     sample = dataset[idx]
#     image = sample["image"].unsqueeze(0).to(DEVICE)
#     sim   = sample["sim"].unsqueeze(0).to(DEVICE)
#     gt_mask = sample["mask"].argmax(dim=0).numpy()

#     click_prob = 1.0
#     round_preds = []
#     round_clicks = []

#     # SIM loop
#     for round_idx in range(SIM_DEPTH):
#         with torch.no_grad():
#             model_input = torch.cat([image, sim], dim=1)
#             logits = model(model_input)
#             pred_prob = torch.softmax(logits[0], dim=0)
#             pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

#         gt_onehot = sample["mask"].to(DEVICE)
#         interaction_mask, click_prob = ddg_virtual_user(
#             pred_prob.cpu(), gt_onehot.cpu(), N_CLASSES, click_prob=click_prob
#         )

#         # Save click coordinates for visualization
#         click_coords = np.argwhere(interaction_mask.numpy() == 1)
#         round_clicks.append(click_coords)
#         round_preds.append(pred)

#         sim[0] = update_sim(sim[0], interaction_mask.to(DEVICE), pred_prob.to(DEVICE), N_CLASSES)

#     # Final prediction
#     with torch.no_grad():
#         final_input = torch.cat([image, sim], dim=1)
#         final_logits = model(final_input)
#         final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

#     # Compute Dice
#     dice_slice = dice_score_per_class(final_pred, gt_mask, N_CLASSES)
#     all_dice_scores.append(dice_slice)

#     # ----------------------
#     # Visualization for selected slices
#     # ----------------------
#     if idx in visualization_indices:
#         ncols = 3 + SIM_DEPTH + 1  # CT + GT + Initial + rounds + Final
#         fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 5))

#         # Original CT
#         axs[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
#         axs[0].set_title("CT Slice")
#         axs[0].axis("off")

#         # Ground Truth
#         axs[1].imshow(gt_mask, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[1].set_title("Ground Truth")
#         axs[1].axis("off")

#         # Initial Prediction
#         axs[2].imshow(round_preds[0], cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[2].set_title("Initial Prediction")
#         axs[2].axis("off")

#         # Predictions after each round
#         for r in range(SIM_DEPTH):
#             axs[3+r].imshow(round_preds[r], cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#             axs[3+r].set_title(f"Round {r+1}")
#             axs[3+r].axis("off")

#             # Mark clicks
#             coords = round_clicks[r]
#             for c, y, x in coords:
#                 axs[3+r].plot(x, y, marker="x", markersize=6,
#                             markeredgewidth=2, color="white", markeredgecolor="black")

#         # Final Prediction
#         axs[-1].imshow(final_pred, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[-1].set_title("Final Prediction")
#         axs[-1].axis("off")

#         # Legend
#         cmap = plt.get_cmap("tab20", N_CLASSES)
#         patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}") for c in range(N_CLASSES)]
#         fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

#         plt.tight_layout()
#         save_path = os.path.join(SAVE_DIR, f"vis_slice_{idx}.png")
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()
#         print(f"   ✅ Saved visualization: {save_path}")

# # ----------------------
# # Compute average Dice per organ
# # ----------------------
# all_dice_scores = np.array(all_dice_scores)
# mean_dice = np.nanmean(all_dice_scores, axis=0)

# print("\n📊 Average Organ-wise Dice Scores (on 30–40 random slices):")
# for i, cls_name in enumerate(CLASSES):
#     print(f"   {cls_name:10s}: {mean_dice[i]:.4f}")

# print("\n✅ Evaluation + Visualization complete!")



import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os
import glob
import shutil

from loader import MedicalSegmentationDataset
# Use the Swin model by default (recommended after your changes). If you want ResNet, change this import.
from models import SwinUNet_ConvLSTM_Enhanced as ModelClass
from ddg_virt_user import ddg_virtual_user
from sim import update_sim

# ----------------------
# Config
# ----------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CLASSES = [f"class_{i}" for i in range(11)]
N_CLASSES = len(CLASSES)
SIM_DEPTH = 3
SAVE_DIR = "results_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 Clear existing files in SAVE_DIR
for f in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, f)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"⚠️ Failed to delete {file_path}. Reason: {e}")

# ----------------------
# Pick 2-3 random patient pickle files
# ----------------------
PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data"
all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))

if len(all_pkls) == 0:
    raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))
chosen_names = [os.path.basename(f) for f in chosen_pkls]
print(f"✅ Selected patient files: {chosen_names}")

# ----------------------
# Load dataset
# ----------------------
dataset = MedicalSegmentationDataset(
    pkl_dir=PKL_DIR,
    classes=CLASSES,        # pass names (safer). If your dataset expects indices, change back to list(range(N_CLASSES))
    sim_depth=SIM_DEPTH,
    load_sim=True,
)

# ----------------------
# Load model
# ----------------------
model = ModelClass(
    num_classes=N_CLASSES,
    num_seg_classes=N_CLASSES,
    sim_depth=SIM_DEPTH,
    use_deconv=True,
    pretrained=False,   # set True if you want to download / use pretrained swin weights
    img_size=256
).to(DEVICE)

# Adjust this checkpoint path to your best model
CHECKPOINT_PATH = "checkpoints/best_swin_ddg.pth"
if not os.path.exists(CHECKPOINT_PATH):
    print(f"⚠️ Checkpoint not found at {CHECKPOINT_PATH}. Continuing with uninitialized weights.")
else:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # handles both raw state_dict and wrapped checkpoint
        model.load_state_dict(checkpoint)
model.eval()

# ----------------------
# Dice function
# ----------------------
def dice_score_per_class(pred, gt, num_classes):
    dice_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        gt_c = (gt == c).astype(np.uint8)
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice = (2 * intersection / union) if union > 0 else np.nan
        dice_per_class[c] = dice
    return dice_per_class

# ----------------------
# Build index of slices for the chosen patients
# ----------------------
all_indices = []
for pkl_name in chosen_names:
    # dataset.slice_index expected elements like (fpath, pid, slice_idx)
    patient_indices = [
        i for i, (fpath, pid, slice_idx) in enumerate(dataset.slice_index)
        if os.path.basename(fpath) == pkl_name
    ]
    if len(patient_indices) == 0:
        print(f"⚠️ No slices found in dataset for {pkl_name}")
    all_indices.extend(patient_indices)

if len(all_indices) == 0:
    raise RuntimeError("No slices found for the selected pkl files in the dataset.")

num_slices = min(30, len(all_indices))
chosen_indices = random.sample(all_indices, num_slices)
print(f"\n🎯 Selected {len(chosen_indices)} total slices for evaluation\n")

# ----------------------
# Loop: run inference, compute Dice, and visualize a few slices
# ----------------------
all_dice_scores = []
visualization_indices = random.sample(chosen_indices, min(6, len(chosen_indices)))  # visualize only few

for idx in chosen_indices:
    sample = dataset[idx]
    # ensure shapes: image (1,H,W), sim (2*D*N, H, W), mask (N,H,W)
    image = sample["image"].unsqueeze(0).to(DEVICE).float()
    image = image / 255.0
    image = (image - 0.5) / 0.5
    sim   = sample["sim"].unsqueeze(0).to(DEVICE)     # (1, 2*D*N, H, W)
    gt_mask = sample["mask"].argmax(dim=0).numpy()   # (H, W)

    click_prob = 1.0
    round_preds = []
    round_clicks = []

    # SIM iterative rounds
    for round_idx in range(SIM_DEPTH):
        with torch.no_grad():
            model_input = torch.cat([image, sim], dim=1)
            logits = model(model_input)                           # (B, N, H, W)
            pred_prob = torch.softmax(logits[0], dim=0)           # (N, H, W)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        gt_onehot = sample["mask"].to(DEVICE)
        interaction_mask, click_prob = ddg_virtual_user(
            pred_prob.cpu(), gt_onehot.cpu(), N_CLASSES, click_prob=click_prob
        )

        # Save click coordinates for visualization (robust to 2D or 3D interaction_mask)
        im_np = interaction_mask.cpu().numpy()
        coords = np.argwhere(im_np == 1)
        # coords can be (H,W) positions (2 cols) or (class,H,W) (3 cols). Normalize to list of (y,x)
        click_list = []
        if coords.size == 0:
            click_list = []
        elif coords.shape[1] == 2:
            # (y, x)
            click_list = [(int(y), int(x)) for y, x in coords]
        elif coords.shape[1] == 3:
            # (class, y, x) -> drop class for plotting
            click_list = [(int(y), int(x)) for _, y, x in coords]
        else:
            # fallback: flatten
            click_list = [(int(r[0]), int(r[1])) for r in coords[:, -2:]]

        round_clicks.append(click_list)
        round_preds.append(pred)

        # update SIM in-place for the first (and subsequent) batch item
        sim[0] = update_sim(sim[0], interaction_mask.to(DEVICE), pred_prob.to(DEVICE), N_CLASSES)

    # Final prediction after all SIM rounds
    with torch.no_grad():
        final_input = torch.cat([image, sim], dim=1)
        final_logits = model(final_input)
        final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

    # Compute Dice
    dice_slice = dice_score_per_class(final_pred, gt_mask, N_CLASSES)
    all_dice_scores.append(dice_slice)

    # ----------------------
    # Visualization for selected slices (guard against too few rounds)
    # ----------------------
    if idx in visualization_indices:
        # number of columns: CT + GT + initial + rounds + Final
        n_rounds_available = max(1, len(round_preds))
        ncols = 3 + n_rounds_available + 1
        fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

        # flatten axs if only one axis returned
        if ncols == 1:
            axs = [axs]

        col = 0
        # Original CT (image is single-channel)
        img_np = image.squeeze().cpu().numpy()
        axs[col].imshow(img_np, cmap="gray")
        axs[col].set_title("CT Slice")
        axs[col].axis("off")
        col += 1

        # Ground Truth
        axs[col].imshow(gt_mask, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
        axs[col].set_title("Ground Truth")
        axs[col].axis("off")
        col += 1

        # Initial Prediction (round_preds[0] if exists)
        init_pred = round_preds[0] if len(round_preds) > 0 else final_pred
        axs[col].imshow(init_pred, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
        axs[col].set_title("Initial Prediction")
        axs[col].axis("off")
        col += 1

        # Predictions after each round (may be same as initial for first)
        for r in range(n_rounds_available):
            pred_r = round_preds[r] if r < len(round_preds) else final_pred
            axs[col].imshow(pred_r, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
            axs[col].set_title(f"Round {r+1}")
            axs[col].axis("off")

            # Mark clicks if present
            coords = round_clicks[r] if r < len(round_clicks) else []
            for (y, x) in coords:
                # protect bounds (rarely necessary)
                H, W = pred_r.shape
                if 0 <= y < H and 0 <= x < W:
                    axs[col].plot(x, y, marker="x", markersize=6, markeredgewidth=2,
                                  markeredgecolor="black", markerfacecolor="white")
            col += 1

        # Final Prediction
        axs[col].imshow(final_pred, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
        axs[col].set_title("Final Prediction")
        axs[col].axis("off")

        # Legend (small)
        cmap = plt.get_cmap("tab20", N_CLASSES)
        patches = [mpatches.Patch(color=cmap(c), label=f"Class {c}") for c in range(N_CLASSES)]
        fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"vis_slice_{idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved visualization: {save_path}")

# ----------------------
# Compute average Dice per organ
# ----------------------
all_dice_scores = np.array(all_dice_scores)
mean_dice = np.nanmean(all_dice_scores, axis=0)

print("\n📊 Average Organ-wise Dice Scores (on selected random slices):")
for i, cls_name in enumerate(CLASSES):
    print(f"   {cls_name:10s}: {mean_dice[i]:.4f}")

print("\n✅ Evaluation + Visualization complete!")
