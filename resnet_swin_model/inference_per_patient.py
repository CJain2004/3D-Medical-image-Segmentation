import os
import glob
import torch
import numpy as np
from loader import MedicalSegmentationDataset
from ddg_virt_user import ddg_virtual_user
from sim import update_sim
from models import SwinUNet_ConvLSTM_SIM

# ---------------------- CONFIG ----------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PKL_DIR = "/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/preprocessed_patients_new_data"
CHECKPOINT_PATH = "checkpoints/best_swin_convlstm_sim.pth"
CLASSES = [f"class_{i}" for i in range(11)]
N_CLASSES = len(CLASSES)
SIM_DEPTH = 4

IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float32, device=DEVICE).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32, device=DEVICE).view(1,3,1,1)

# ---------------------- FUNCTIONS ----------------------

def dice_per_class(pred, gt, num_classes):
    dice = np.full(num_classes, np.nan, dtype=float)
    for c in range(num_classes):
        p = (pred == c).astype(np.uint8)
        g = (gt == c).astype(np.uint8)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        dice[c] = (2.0 * inter / denom) if denom > 0 else np.nan
    return dice


def evaluate_single_patient(pkl_name, dataset, model):
    """Evaluate all slices for a single PKL file and return mean Dice per class."""
    # Collect slice indices belonging to this patient
    indices = [i for i, (fpath, _, _) in enumerate(dataset.slice_index)
               if os.path.basename(fpath) == os.path.basename(pkl_name)]
    if not indices:
        print(f"⚠️ No slices found for {os.path.basename(pkl_name)}")
        return None

    all_dice = []

    for idx in indices:
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(DEVICE).float()
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        img = img / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        sim = sample["sim"].unsqueeze(0).to(DEVICE).float()
        gt_label_map = sample["mask"].argmax(dim=0).cpu().numpy()

        with torch.no_grad():
            state = None
            click_prob = 1.0
            for r in range(SIM_DEPTH):
                logits, state = model(img, sim, state)
                probs_device = torch.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs_device, dim=0).cpu().numpy()

                interaction_mask, click_prob = ddg_virtual_user(
                    pred_prob=probs_device,
                    gt_mask=sample["mask"],
                    num_classes=N_CLASSES,
                    click_prob=click_prob,
                    max_clicks=3,
                    drop_rate=0.5
                )
                sim[0] = update_sim(sim[0], interaction_mask, probs_device, N_CLASSES)

            # Final prediction
            final_logits, _ = model(img, sim, state)
            final_pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

        dice = dice_per_class(final_pred, gt_label_map, N_CLASSES)
        all_dice.append(dice)

    all_dice = np.array(all_dice)
    mean_dice = np.nanmean(all_dice, axis=0)
    return mean_dice


# ---------------------- MAIN EVAL ----------------------
def main():
    all_pkls = glob.glob(os.path.join(PKL_DIR, "*.pkl"))
    if not all_pkls:
        raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

    # choose subset of patients (e.g. 5)
    chosen_pkls = all_pkls[:5]
    print(f"🩻 Evaluating {len(chosen_pkls)} patients:\n" +
          "\n".join([" - " + os.path.basename(p) for p in chosen_pkls]))

    dataset = MedicalSegmentationDataset(
        pkl_dir=PKL_DIR,
        classes=CLASSES,
        sim_depth=SIM_DEPTH,
        load_sim=True
    )

    # load model
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
        print(f"⚠️ Checkpoint not found. Using random weights.")
    model.eval()

    # evaluate each patient
    patient_dices = []
    for pkl_path in chosen_pkls:
        print(f"\n🔍 Evaluating patient: {os.path.basename(pkl_path)} ...")
        mean_dice = evaluate_single_patient(pkl_path, dataset, model)
        if mean_dice is not None:
            patient_dices.append(mean_dice)
            for i, cls_name in enumerate(CLASSES):
                print(f"  {cls_name:10s}: {mean_dice[i]:.4f}")

    # overall summary
    if patient_dices:
        patient_dices = np.array(patient_dices)
        overall_mean = np.nanmean(patient_dices, axis=0)
        print("\n📊 ===== Overall Average Dice Across Patients =====")
        for i, cls_name in enumerate(CLASSES):
            print(f"  {cls_name:10s}: {overall_mean[i]:.4f}")
        print("✅ Evaluation Completed Successfully")


if __name__ == "__main__":
    main()
