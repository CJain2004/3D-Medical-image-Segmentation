import os
import pickle
import numpy as np

pkl_dir = "processed_patients_ras"          # your current pickle folder
npz_dir = "processed_patients_ras_npz"      # new folder for compressed files
os.makedirs(npz_dir, exist_ok=True)

for fname in os.listdir(pkl_dir):
    if not fname.endswith(".pkl"):
        continue

    pkl_path = os.path.join(pkl_dir, fname)
    npz_path = os.path.join(npz_dir, fname.replace(".pkl", ".npz"))

    # ✅ Skip if already converted
    if os.path.exists(npz_path):
        print(f"Skipping {fname}, already converted")
        continue

    try:
        # Load from pickle
        with open(pkl_path, "rb") as f:
            ct_data, mask_data, voxel_spacing = pickle.load(f)

        # Save compressed
        np.savez_compressed(npz_path, ct=ct_data, mask=mask_data, spacing=voxel_spacing)

        print(f"✅ Converted {fname} → {npz_path}")

    except Exception as e:
        print(f"❌ Error converting {fname}: {e}")
