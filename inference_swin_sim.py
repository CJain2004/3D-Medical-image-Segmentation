# # check_model_swin_sim.py
# import os
# import glob
# import random
# import shutil
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# from loader import MedicalSegmentationDataset
# from ddg_virt_user import ddg_virtual_user
# from sim import update_sim

# # Try importing several possible model names exported from your models.py
# try:
#     from models import SwinUNet_ConvLSTM_Enhanced as ModelClass
# except Exception:
#     try:
#         from models import SwinUNet_ConvLSTM_SIM as ModelClass
#     except Exception:
#         try:
#             from models import Swin_ConvLSTM_UNet_SIM as ModelClass
#         except Exception as e:
#             raise ImportError("Could not import a Swin+ConvLSTM model from models.py. "
#                               "Export one of: SwinUNet_ConvLSTM_Enhanced, SwinUNet_ConvLSTM_SIM, Swin_ConvLSTM_UNet_SIM") from e

# # ----------------------
# # CONFIG
# # ----------------------
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_test_data"
# SAVE_DIR = "results_vis"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # clear previous visualizations
# for f in os.listdir(SAVE_DIR):
#     path = os.path.join(SAVE_DIR, f)
#     try:
#         if os.path.isfile(path) or os.path.islink(path):
#             os.unlink(path)
#         elif os.path.isdir(path):
#             shutil.rmtree(path)
#     except Exception:
#         pass

# # classes / SIM config - update if different
# CLASSES = [f"class_{i}" for i in range(11)]
# N_CLASSES = len(CLASSES)
# SIM_DEPTH = 3

# # checkpoint (change if needed)
# CHECKPOINT_PATH = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/checkpoints/best_swin_convlstm_sim.pth"

# # ----------------------
# # Sanity checks & dataset
# # ----------------------
# all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))
# if len(all_pkls) == 0:
#     raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

# chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))
# chosen_names = [os.path.basename(p) for p in chosen_pkls]
# print("Selected patient files:", chosen_names)

# dataset = MedicalSegmentationDataset(pkl_dir=PKL_DIR, classes=CLASSES, sim_depth=SIM_DEPTH, load_sim=True)
# print(f"Dataset loaded: {len(dataset)} slices available")

# # collect indices corresponding to chosen pkls
# all_indices = []
# for pkl_name in chosen_names:
#     inds = [i for i, (fpath, pid, sl) in enumerate(dataset.slice_index) if os.path.basename(fpath) == pkl_name]
#     if len(inds) == 0:
#         print(f"Warning: no slices found for {pkl_name} in dataset.slice_index")
#     all_indices.extend(inds)

# if len(all_indices) == 0:
#     raise RuntimeError("No slices found for chosen pkls in dataset")

# # sample slices to evaluate
# num_slices = min(300000, len(all_indices))
# chosen_indices = random.sample(all_indices, num_slices)
# print(f"Evaluating {len(chosen_indices)} slices (random subset)")

# # ----------------------
# # Load model
# # ----------------------
# # instantiate model with sensible defaults (adjust if your model signature differs)
# model = ModelClass(num_classes=N_CLASSES, num_seg_classes=N_CLASSES, sim_depth=SIM_DEPTH, pretrained=False).to(DEVICE)
# if os.path.exists(CHECKPOINT_PATH):
#     ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     if isinstance(ck, dict) and "model_state_dict" in ck:
#         model.load_state_dict(ck["model_state_dict"], strict=False)
#     else:
#         model.load_state_dict(ck, strict=False)
#     print("Loaded checkpoint:", CHECKPOINT_PATH)
# else:
#     print("No checkpoint found at", CHECKPOINT_PATH, "- running with random weights")

# model.eval()

# # ----------------------
# # Utility: per-class Dice
# # ----------------------
# def dice_score_per_class(pred, gt, num_classes):
#     dice = np.full(num_classes, np.nan, dtype=float)
#     for c in range(num_classes):
#         p = (pred == c).astype(np.uint8)
#         g = (gt == c).astype(np.uint8)
#         inter = (p & g).sum()
#         denom = p.sum() + g.sum()
#         dice[c] = (2.0 * inter / denom) if denom > 0 else np.nan
#     return dice

# # ----------------------
# # Inference loop
# # ----------------------
# all_dice = []
# vis_indices = set(random.sample(chosen_indices, min(6, len(chosen_indices))))

# for idx in chosen_indices:
#     sample = dataset[idx]
#     # image: (1,H,W) -> to (1,C,H,W)
#     img = sample["image"].unsqueeze(0).to(DEVICE).float()  # (1,1,H,W) or (1,3,H,W)
#     # normalize same as training: /255 then (x-0.5)/0.5
#     img = img / 255.0
#     img = (img - 0.5) / 0.5
#     if img.shape[1] == 1:
#         img = img.repeat(1, 3, 1, 1)  # replicate grayscale to 3 channels for Swin

#     sim = sample["sim"].unsqueeze(0).to(DEVICE).float()  # (1, 2*D*N, H, W)
#     gt_mask = sample["mask"].argmax(dim=0).cpu().numpy()  # (H,W)

#     click_prob = 1.0
#     round_preds = []
#     round_clicks = []

#     # iterative SIM rounds
#     with torch.no_grad():
#         state = None
#         for r in range(SIM_DEPTH):
#             # forward: model may accept (img, sim) or (img, sim, state). handle both
#             try:
#                 logits, state = model(img, sim, state)
#             except TypeError:
#                 # model signature might be (input_tensor) expecting concatenated input
#                 inp = torch.cat([img, sim], dim=1)
#                 logits = model(inp)
#                 state = None
#             probs = torch.softmax(logits, dim=1)[0].cpu()  # (N, H, W)
#             pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

#             # simulate clicks (use cpu tensors for ddg)
#             inter_mask, click_prob = ddg_virtual_user(pred_prob=probs.clone(), gt_mask=sample["mask"].cpu(), num_classes=N_CLASSES, click_prob=click_prob)
#             # record click coordinates for visualization
#             coords = np.argwhere(inter_mask.cpu().numpy() == 1)
#             click_list = []
#             if coords.size != 0:
#                 if coords.shape[1] == 2:
#                     click_list = [(int(y), int(x)) for y, x in coords]
#                 elif coords.shape[1] == 3:
#                     click_list = [(int(y), int(x)) for _, y, x in coords]
#                 else:
#                     click_list = [(int(r[-2]), int(r[-1])) for r in coords]
#             round_clicks.append(click_list)
#             round_preds.append(pred)

#             # update SIM in-place for next round (except after last round)
#             if r < SIM_DEPTH - 1:
#                 sim[0] = update_sim(sim[0], inter_mask.to(DEVICE), probs.to(DEVICE), N_CLASSES)

#         # final forward (after all rounds) to get final_pred
#         try:
#             final_logits, _ = model(img, sim, state)
#         except TypeError:
#             final_logits = model(torch.cat([img, sim], dim=1))
#         final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

#     # compute dice
#     dice = dice_score_per_class(final_pred, gt_mask, N_CLASSES)
#     all_dice.append(dice)

#     # visualization if selected
#     if idx in vis_indices:
#         n_rounds_avail = len(round_preds)
#         ncols = 3 + n_rounds_avail + 1
#         fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
#         col = 0

#         # CT
#         img_np = (img[0].cpu().numpy().transpose(1,2,0))
#         if img_np.shape[2] == 3:
#             img_disp = img_np[:,:,0]  # show one channel (grayscale rep)
#         else:
#             img_disp = img_np
#         axs[col].imshow(img_disp, cmap="gray")
#         axs[col].set_title("CT Slice")
#         axs[col].axis("off")
#         col += 1

#         # GT
#         axs[col].imshow(gt_mask, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[col].set_title("Ground Truth")
#         axs[col].axis("off")
#         col += 1

#         # initial pred
#         init_pred = round_preds[0] if n_rounds_avail > 0 else final_pred
#         axs[col].imshow(init_pred, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[col].set_title("Initial Prediction")
#         axs[col].axis("off")
#         col += 1

#         # predictions per round
#         for r in range(n_rounds_avail):
#             axs[col].imshow(round_preds[r], cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#             axs[col].set_title(f"Round {r+1}")
#             axs[col].axis("off")
#             # plot clicks
#             for (y, x) in round_clicks[r]:
#                 H, W = round_preds[r].shape
#                 if 0 <= y < H and 0 <= x < W:
#                     axs[col].plot(x, y, marker="x", markersize=6, markeredgewidth=2, color="black")
#             col += 1

#         # final
#         axs[col].imshow(final_pred, cmap="tab20", vmin=0, vmax=N_CLASSES-1)
#         axs[col].set_title("Final Prediction")
#         axs[col].axis("off")

#         # legend
#         cmap = plt.get_cmap("tab20", N_CLASSES)
#         patches = [mpatches.Patch(color=cmap(i), label=f"{i}") for i in range(N_CLASSES)]
#         fig.legend(handles=patches, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))

#         plt.tight_layout()
#         out_path = os.path.join(SAVE_DIR, f"vis_slice_{idx}.png")
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()
#         print("Saved visualization:", out_path)

# # ----------------------
# # Summary Dice
# # ----------------------
# all_dice = np.array(all_dice)  # (num_slices, N_CLASSES)
# mean_dice = np.nanmean(all_dice, axis=0)

# print("\nAverage Dice per class (over evaluated slices):")
# for i, name in enumerate(CLASSES):
#     print(f"  {name:12s}: {mean_dice[i]:.4f}")

# print("\nDone.")


# check_model_swin_sim_new.py
import os
import glob
import random
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from loader import MedicalSegmentationDataset
from ddg_virt_user import ddg_virtual_user
from sim import update_sim
from models import SwinUNet_ConvLSTM_SIM
import imageio

# ----------------------
# CONFIG
# ----------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_test_data"
SAVE_DIR = "results_vis"
os.makedirs(SAVE_DIR, exist_ok=True)
# make sure IMAGENET_MEAN/STD are defined and on DEVICE
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float32, device=DEVICE).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32, device=DEVICE).view(1,3,1,1)

def create_interaction_gif(img_np, gt_mask, round_preds, round_clicks, final_pred, save_path):
    frames = []
    durations = []   # seconds per frame

    # helper to capture current figure as RGB uint8
    def grab_fig_as_rgb(fig):
        fig.canvas.draw()
        arr = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return arr[..., :3].astype(np.uint8)  # drop alpha -> RGB

    # 1. Show original CT slice
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_np, cmap="gray")
    ax.set_title("CT Slice")
    ax.axis("off")
    frames.append(grab_fig_as_rgb(fig))
    durations.append(5000.0)   # stay 5 seconds

    # 2. Show Ground Truth (optional long or short)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(gt_mask, cmap="tab20")
    ax.set_title("Ground Truth")
    ax.axis("off")
    frames.append(grab_fig_as_rgb(fig))
    durations.append(5000.0)   # e.g. 2 seconds for GT

    # 3. Round-by-round animation
    for r, pred in enumerate(round_preds):
        # prediction frame (long)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(pred, cmap="tab20", vmin=0, vmax=10)
        ax.set_title(f"Round {r+1} – Model Prediction")
        ax.axis("off")
        frames.append(grab_fig_as_rgb(fig))
        durations.append(5000.0)   # stay 5 seconds for prediction

        # clicks overlay frame (shorter)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(pred, cmap="tab20", vmin=0, vmax=10)
        for (y, x) in round_clicks[r]:
            ax.plot(x, y, "x", color="red", markersize=10, markeredgewidth=2)
        ax.set_title(f"Round {r+1} – User Clicks")
        ax.axis("off")
        frames.append(grab_fig_as_rgb(fig))
        durations.append(5000.0)   # 1 second for clicks overlay

    # 4. Final Prediction
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(final_pred, cmap="tab20", vmin=0, vmax=10)
    ax.set_title("Final Prediction")
    ax.axis("off")
    frames.append(grab_fig_as_rgb(fig))
    durations.append(5000.0)   # 5 seconds

    # Write GIF with per-frame durations
    imageio.mimsave(save_path, frames, duration=durations)
    print("🎞️ Saved GIF:", save_path)


# clear previous visualizations
for f in os.listdir(SAVE_DIR):
    path = os.path.join(SAVE_DIR, f)
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception:
        pass

# dataset / model params
CLASSES = [
    "Background",   # 0
    "Esophagus",    # 1
    "Lung_L",       # 2
    "Lung_R",       # 3
    "Kidney_L",     # 4
    "Kidney_R",     # 5
    "Stomach",      # 6
    "Liver",        # 7
    "Spleen",       # 8
    "Pancreas",     # 9
    "Intestine"     # 10
]
N_CLASSES = len(CLASSES)
SIM_DEPTH = 4
CHECKPOINT_PATH = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/checkpoints/best_swin_convlstm_sim_depth_4.pth"

# ----------------------
# Load dataset
# ----------------------
all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))
if not all_pkls:
    raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")
print("Evaluating all PKL files found:", len(all_pkls))

chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))
# chosen_pkls = all_pkls
# print("Selected patient files:", [os.path.basename(p) for p in chosen_pkls])

dataset = MedicalSegmentationDataset(
    pkl_dir=PKL_DIR,
    classes=CLASSES,
    sim_depth=SIM_DEPTH,
    load_sim=True
)
print(f"Dataset loaded: {len(dataset)} slices")

# collect indices for chosen pkls
indices = []
for pkl_name in chosen_pkls:
    idx = [i for i, (fpath, _, _) in enumerate(dataset.slice_index)
           if os.path.basename(fpath) == os.path.basename(pkl_name)]
    indices.extend(idx)
if not indices:
    raise RuntimeError("No slices found for chosen PKL files")

# chosen_indices = random.sample(indices, min(200, len(indices)))
chosen_indices = list(range(len(dataset)))

print(f"Evaluating {len(chosen_indices)} slices")

# ----------------------
# Load model
# ----------------------
model = SwinUNet_ConvLSTM_SIM(
    num_classes=N_CLASSES,
    num_seg_classes=N_CLASSES,
    sim_depth=SIM_DEPTH,
    pretrained=False
).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"⚠️ No checkpoint found at {CHECKPOINT_PATH}, using random weights.")

model.eval()

# ----------------------
# Dice score
# ----------------------
def dice_per_class(pred, gt, num_classes):
    dice = np.full(num_classes, np.nan, dtype=float)
    for c in range(num_classes):
        p = (pred == c).astype(np.uint8)
        g = (gt == c).astype(np.uint8)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        dice[c] = (2.0 * inter / denom) if denom > 0 else np.nan
    return dice

# ----------------------
# Inference loop
# ----------------------
all_dice = []
vis_indices = set(random.sample(chosen_indices, min(6, len(chosen_indices))))

for idx in vis_indices:
    sample = dataset[idx]

    img = sample["image"].unsqueeze(0).to(DEVICE).float()
    # if img.shape[1] == 1:
    #     img = img.repeat(1, 3, 1, 1)
    img = img / 255.0
    mean = img.mean(dim=(2, 3), keepdim=True)
    std = img.std(dim=(2, 3), keepdim=True) + 1e-5
    img = (img - mean) / std

    # img = img / 255.0
    # img = (img - IMAGENET_MEAN) / IMAGENET_STD

    sim = sample["sim"].unsqueeze(0).to(DEVICE).float()   # (1, sim_ch, H, W)
    gt_label_map = sample["mask"].argmax(dim=0).cpu().numpy()  # (H, W) numpy for class lookup
    H, W = gt_label_map.shape

    with torch.no_grad():
        state = None
        round_preds, round_clicks = [], []
        click_prob=1.0
        # multi-round interaction loop
        for r in range(SIM_DEPTH):
            logits, state = model(img, sim, state)                # logits: (1, C, H, W)
            probs_device = torch.softmax(logits, dim=1)[0]       # (C, H, W) on DEVICE
            pred = torch.argmax(probs_device, dim=0).cpu().numpy()  # (H, W) numpy

            # store prediction
            round_preds.append(pred)

            # Generate user interactions (binary click mask)
            interaction_mask, click_prob = ddg_virtual_user(
                pred_prob=probs_device,
                gt_mask=sample["mask"],
                num_classes=N_CLASSES,
                click_prob=click_prob,
                max_clicks=6,
                drop_rate=0.4
            )

            # ---- Extract (y, x) click coordinates for visualization ----
            clicks_coords = []
            mask_cpu = interaction_mask.detach().cpu().numpy()

            # Loop through each class to collect click pixel positions
            for cls in range(mask_cpu.shape[0]):
                ys, xs = np.where(mask_cpu[cls] == 1)
                for (y, x) in zip(ys, xs):
                    clicks_coords.append((int(y), int(x)))

            # Save click coordinates for this round (for plotting later)
            round_clicks.append(clicks_coords)

            # ---- Update SIM for next iteration ----
            sim[0] = update_sim(sim[0], interaction_mask, probs_device, N_CLASSES)


        # final forward after all rounds
        final_logits, _ = model(img, sim, state)
        final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

    dice = dice_per_class(final_pred, gt_label_map, N_CLASSES)
    all_dice.append(dice)
    if idx == list(vis_indices)[0]:    # only first image
        img_np = sample["image"].squeeze(0).cpu().numpy()

        create_interaction_gif(
            img_np=img_np,
            gt_mask=gt_label_map,
            round_preds=round_preds,
            round_clicks=round_clicks,
            final_pred=final_pred,
            save_path=os.path.join(SAVE_DIR, f"anim_slice_{idx}.gif")
        )

    # ---------------- Visualization (fixed) ----------------
    if idx in vis_indices:
        # defensive: ensure we have preds/clicks
        rounds = len(round_preds)
        if rounds == 0:
            print(f"Warning: no round predictions for idx={idx}, skipping visualization.")
        else:
            ncols = 3 + rounds + 1
            fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
            col = 0

            # original (unnormalized) image for display
            
            vis_img = sample["image"].squeeze(0).cpu().numpy()  # (H, W) for grayscale
            axs[col].imshow(vis_img, cmap="gray")
            axs[col].set_title("CT Slice")
            axs[col].axis("off")
            col += 1

            # ground truth (use gt_label_map which you built earlier)
            axs[col].imshow(gt_label_map, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
            axs[col].set_title("Ground Truth")
            axs[col].axis("off")
            col += 1

            # initial prediction (first element of round_preds)
            axs[col].imshow(round_preds[0], cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
            axs[col].set_title("Initial Prediction")
            axs[col].axis("off")
            col += 1

            # per-round preds and click markers
            for r in range(rounds):
                axs[col].imshow(round_preds[r], cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
                axs[col].set_title(f"Round {r+1}")
                axs[col].axis("off")
                # plot any clicks (if present)
                if r < len(round_clicks):
                    for (y, x) in round_clicks[r]:
                        axs[col].plot(x, y, "x", color="black", markersize=5, markeredgewidth=1.5)
                col += 1

            # final prediction
            axs[col].imshow(final_pred, cmap="tab20", vmin=0, vmax=N_CLASSES - 1)
            axs[col].set_title("Final Prediction")
            axs[col].axis("off")

            cmap = plt.get_cmap("tab20", N_CLASSES)
            patches = [mpatches.Patch(color=cmap(i), label=f"{i}") for i in range(N_CLASSES)]
            fig.legend(handles=patches, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            out_path = os.path.join(SAVE_DIR, f"vis_slice_{idx}.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"🖼️ Saved visualization: {out_path}")


# ----------------------
# Dice Summary
# ----------------------
all_dice = np.array(all_dice)
mean_dice = np.nanmean(all_dice, axis=0)

print("\n📊 Average Dice per class:")
for i, name in enumerate(CLASSES):
    print(f"  {name:12s}: {mean_dice[i]:.4f}")

print("\n✅ Evaluation completed successfully.")



# # check_model_swin_sim_new.py
# import os
# import glob
# import random
# import shutil
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# from loader import MedicalSegmentationDataset
# from ddg_virt_user import ddg_virtual_user
# from sim import update_sim
# from models import SwinUNet_ConvLSTM_SIM
# import imageio

# # ----------------------
# # CONFIG
# # ----------------------
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_test_data"
# SAVE_DIR = "results_vis"
# os.makedirs(SAVE_DIR, exist_ok=True)

# IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float32, device=DEVICE).view(1,3,1,1)
# IMAGENET_STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32, device=DEVICE).view(1,3,1,1)


# def create_interaction_gif(img_np, gt_mask, round_preds, round_clicks, final_pred, save_path):
#     frames = []
#     durations = []

#     def grab_fig_as_rgb(fig):
#         fig.canvas.draw()
#         arr = np.array(fig.canvas.renderer.buffer_rgba())
#         plt.close(fig)
#         return arr[..., :3].astype(np.uint8)

#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.imshow(img_np, cmap="gray")
#     ax.set_title("CT Slice")
#     ax.axis("off")
#     frames.append(grab_fig_as_rgb(fig))
#     durations.append(5000.0)

#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.imshow(gt_mask, cmap="tab20")
#     ax.set_title("Ground Truth")
#     ax.axis("off")
#     frames.append(grab_fig_as_rgb(fig))
#     durations.append(5000.0)

#     for r, pred in enumerate(round_preds):

#         # prediction frame
#         fig, ax = plt.subplots(figsize=(5,5))
#         ax.imshow(pred, cmap="tab20", vmin=0, vmax=10)

#         if r == 0:
#             ax.set_title("Initial Prediction")
#         else:
#             ax.set_title(f"Round {r} – Model Prediction")

#         ax.axis("off")
#         frames.append(grab_fig_as_rgb(fig))
#         durations.append(5000.0)

#         # clicks frame (only for rounds ≥ 1)
#         if r > 0:
#             fig, ax = plt.subplots(figsize=(5,5))
#             ax.imshow(pred, cmap="tab20", vmin=0, vmax=10)
#             for (y, x) in round_clicks[r-1]:
#                 ax.plot(x, y, "x", color="red", markersize=10, markeredgewidth=2)
#             ax.set_title(f"Round {r} – User Clicks")
#             ax.axis("off")
#             frames.append(grab_fig_as_rgb(fig))
#             durations.append(5000.0)

#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.imshow(final_pred, cmap="tab20", vmin=0, vmax=10)
#     ax.set_title("Final Prediction")
#     ax.axis("off")
#     frames.append(grab_fig_as_rgb(fig))
#     durations.append(5000.0)

#     imageio.mimsave(save_path, frames, duration=durations)
#     print("🎞️ Saved GIF:", save_path)


# # clear old visualizations
# for f in os.listdir(SAVE_DIR):
#     path = os.path.join(SAVE_DIR, f)
#     try:
#         if os.path.isfile(path) or os.path.islink(path):
#             os.unlink(path)
#         elif os.path.isdir(path):
#             shutil.rmtree(path)
#     except Exception:
#         pass

# CLASSES = [
#     "Background", "Esophagus", "Lung_L", "Lung_R",
#     "Kidney_L", "Kidney_R", "Stomach", "Liver",
#     "Spleen", "Pancreas", "Intestine"
# ]
# N_CLASSES = len(CLASSES)
# SIM_DEPTH = 4
# CHECKPOINT_PATH = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/checkpoints/best_swin_convlstm_sim_depth_4.pth"

# all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))
# if not all_pkls:
#     raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")
# print("Evaluating all PKL files found:", len(all_pkls))

# chosen_pkls = random.sample(all_pkls, min(3, len(all_pkls)))

# dataset = MedicalSegmentationDataset(
#     pkl_dir=PKL_DIR,
#     classes=CLASSES,
#     sim_depth=SIM_DEPTH,
#     load_sim=True
# )
# print(f"Dataset loaded: {len(dataset)} slices")

# indices = []
# for pkl_name in chosen_pkls:
#     idx = [i for i, (fpath, _, _) in enumerate(dataset.slice_index)
#            if os.path.basename(fpath) == os.path.basename(pkl_name)]
#     indices.extend(idx)

# chosen_indices = list(range(len(dataset)))
# print(f"Evaluating {len(chosen_indices)} slices")

# # ----------------------
# # Load model
# # ----------------------
# model = SwinUNet_ConvLSTM_SIM(
#     num_classes=N_CLASSES,
#     num_seg_classes=N_CLASSES,
#     sim_depth=SIM_DEPTH,
#     pretrained=False
# ).to(DEVICE)

# if os.path.exists(CHECKPOINT_PATH):
#     state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     model.load_state_dict(state_dict, strict=False)
#     print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
# else:
#     print(f"⚠️ No checkpoint found at {CHECKPOINT_PATH}, using random weights.")

# model.eval()


# # ----------------------
# # Dice
# # ----------------------
# def dice_per_class(pred, gt, num_classes):
#     dice = np.full(num_classes, np.nan, dtype=float)
#     for c in range(num_classes):
#         p = (pred == c).astype(np.uint8)
#         g = (gt == c).astype(np.uint8)
#         inter = (p & g).sum()
#         denom = p.sum() + g.sum()
#         dice[c] = (2.0 * inter / denom) if denom > 0 else np.nan
#     return dice


# # ⭐ ADDED: Track dice per round (Initial + SIM rounds + Final)
# dice_per_round = [[] for _ in range(SIM_DEPTH + 2)]  
# # indices: 0 = initial, 1..SIM_DEPTH = rounds, SIM_DEPTH+1 = final


# # ----------------------
# # Inference loop
# # ----------------------
# all_dice = []
# vis_indices = set(random.sample(chosen_indices, min(6, len(chosen_indices))))

# for idx in chosen_indices:
#     sample = dataset[idx]

#     img = sample["image"].unsqueeze(0).to(DEVICE).float()
#     img = img / 255.0
#     mean = img.mean(dim=(2, 3), keepdim=True)
#     std = img.std(dim=(2, 3), keepdim=True) + 1e-5
#     img = (img - mean) / std

#     sim = sample["sim"].unsqueeze(0).to(DEVICE).float()
#     gt_label_map = sample["mask"].argmax(dim=0).cpu().numpy()
#     H, W = gt_label_map.shape

#     with torch.no_grad():
#         # ⭐ ADDED: Initial prediction BEFORE interactions
#         logits_init, state_init = model(img, sim, None)
#         probs_init = torch.softmax(logits_init, dim=1)[0]
#         pred_init = torch.argmax(probs_init, dim=0).cpu().numpy()

#         round_preds = [pred_init]
#         round_clicks = []
#         click_prob = 1.0

#         # store dice for initial prediction
#         dice_init = dice_per_class(pred_init, gt_label_map, N_CLASSES)
#         dice_per_round[0].append(dice_init)

#         state = state_init

#         # ---- Multi-round ----
#         for r in range(SIM_DEPTH):

#             logits, state = model(img, sim, state)
#             probs_device = torch.softmax(logits, dim=1)[0]
#             pred = torch.argmax(probs_device, dim=0).cpu().numpy()

#             round_preds.append(pred)

#             # ⭐ ADDED: store dice for round r+1
#             dice_r = dice_per_class(pred, gt_label_map, N_CLASSES)
#             dice_per_round[r+1].append(dice_r)

#             interaction_mask, click_prob = ddg_virtual_user(
#                 pred_prob=probs_device,
#                 gt_mask=sample["mask"],
#                 num_classes=N_CLASSES,
#                 click_prob=click_prob,
#                 max_clicks=6,
#                 drop_rate=0.4
#             )

#             clicks_coords = []
#             mask_cpu = interaction_mask.detach().cpu().numpy()
#             for cls in range(mask_cpu.shape[0]):
#                 ys, xs = np.where(mask_cpu[cls] == 1)
#                 for (y, x) in zip(ys, xs):
#                     clicks_coords.append((int(y), int(x)))

#             round_clicks.append(clicks_coords)

#             sim[0] = update_sim(sim[0], interaction_mask, probs_device, N_CLASSES)

#         # ---- Final prediction ----
#         final_logits, _ = model(img, sim, state)
#         final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

#         # ⭐ ADDED: final dice
#         dice_final = dice_per_class(final_pred, gt_label_map, N_CLASSES)
#         dice_per_round[SIM_DEPTH + 1].append(dice_final)

#     # store dice for normal summary
#     all_dice.append(dice_final)

#     # GIF generation (unchanged)
#     if idx == list(vis_indices)[0]:
#         img_np = sample["image"].squeeze(0).cpu().numpy()

#         create_interaction_gif(
#             img_np=img_np,
#             gt_mask=gt_label_map,
#             round_preds=round_preds,
#             round_clicks=round_clicks,
#             final_pred=final_pred,
#             save_path=os.path.join(SAVE_DIR, f"anim_slice_{idx}.gif")
#         )


# # ----------------------
# # Dice Summary (Final only)
# # ----------------------
# all_dice = np.array(all_dice)
# mean_dice = np.nanmean(all_dice, axis=0)

# print("\n📊 Average Dice per class (FINAL only):")
# for i, name in enumerate(CLASSES):
#     print(f"  {name:12s}: {mean_dice[i]:.4f}")


# # ----------------------
# # ⭐ Dice summary per ROUND
# # ----------------------
# print("\n📈 Dice per round (Average over slices):")
# round_names = ["Initial"] + [f"Round {i+1}" for i in range(SIM_DEPTH)] + ["Final"]

# for r in range(SIM_DEPTH + 2):
#     arr = np.array(dice_per_round[r])
#     mean_r = np.nanmean(arr, axis=0)

#     print(f"\n--- {round_names[r]} ---")
#     for c, name in enumerate(CLASSES):
#         print(f"{name:12s}: {mean_r[c]:.4f}")

# print("\n✅ Evaluation completed successfully.")
