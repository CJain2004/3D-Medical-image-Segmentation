import os
import glob
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from nibabel import as_closest_canonical
import pickle

class MedicalDataset:
    def __init__(self, dataset_root, output_dir="processed_patients"):
        """
        dataset_root: path to the folder containing BDMAP_* patient folders
        output_dir: where to save processed .pkl files
        """
        self.dataset_root = dataset_root
        self.patient_dirs = sorted(glob.glob(os.path.join(dataset_root, "BDMAP_*")))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_and_save_all(self):
        """Process each patient and save immediately to disk"""
        for patient_path in self.patient_dirs:
            patient_id = os.path.basename(patient_path)
            out_file = os.path.join(self.output_dir, f"{patient_id}.pkl")

            # ✅ Skip if already processed
            if os.path.exists(out_file):
                print(f"Skipping {patient_id} (already processed)")
                continue

            ct_path = os.path.join(patient_path, "ct.nii.gz")
            mask_path = os.path.join(patient_path, "combined_labels.nii.gz")

            if not os.path.exists(ct_path) or not os.path.exists(mask_path):
                print(f"Skipping {patient_id} - Missing files")
                continue

            try:
                # --- Load CT scan ---
                ct_img = nib.load(ct_path)
                ct_img = as_closest_canonical(ct_img)  # Reorient
                ct_data = ct_img.get_fdata().astype(np.float32)
                voxel_spacing = ct_img.header.get_zooms()

                # --- Load mask ---
                mask_img = nib.load(mask_path)
                mask_img = as_closest_canonical(mask_img)
                mask_data = mask_img.get_fdata().astype(np.uint8)

                # --- Normalize CT values ---
                ct_data = (ct_data - np.min(ct_data)) / (np.max(ct_data) - np.min(ct_data))

                # Save immediately as pickle
                with open(out_file, "wb") as f:
                    pickle.dump((ct_data, mask_data, voxel_spacing), f)

                print(f"✅ Saved {patient_id} → {out_file}")

            except Exception as e:
                print(f"❌ Error processing {patient_id}: {e}")

if __name__ == "__main__":
    dataset_path = r"/home/Drivessd2tb/cherish/Interactive_segmentation/Data_prep/AbdomenAtlas1.0Mini"
    med_dataset = MedicalDataset(dataset_path, output_dir="processed_patients_ras")
    med_dataset.process_and_save_all()
