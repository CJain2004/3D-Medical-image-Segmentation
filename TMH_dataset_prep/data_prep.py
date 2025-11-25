import os
import glob
import numpy as np
import pickle

class NumpyMedicalDataset:
    def __init__(self, dataset_root, output_dir="processed_np_patients"):
        """
        dataset_root: path to folder containing Auto_seg_* patient folders
        output_dir: where to save processed .pkl files
        """
        self.dataset_root = dataset_root
        self.patient_dirs = sorted(glob.glob(os.path.join(dataset_root, "Auto_seg_*")))
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

            npz_files = sorted(glob.glob(os.path.join(patient_path, "*.npz")))
            if not npz_files:
                print(f"Skipping {patient_id} - No .npz files found")
                continue

            try:
                all_imgs = []
                all_gts = []

                for f in npz_files:
                    data = np.load(f)
                    imgs = data['image']
                    gts = data['label']

                    # Ensure float32 for CT, uint8 for mask
                    imgs = imgs.astype(np.float32)
                    gts = gts.astype(np.uint8)

                    all_imgs.append(imgs)
                    all_gts.append(gts)

                # Stack all slices into a 3D volume
                ct_data = np.stack(all_imgs, axis=0)  # shape: (num_slices, H, W)
                mask_data = np.stack(all_gts, axis=0)

                # --- Normalize CT values per patient ---
                ct_min, ct_max = ct_data.min(), ct_data.max()
                if ct_max > ct_min:  # avoid divide-by-zero
                    ct_data = (ct_data - ct_min) / (ct_max - ct_min)

                # Save immediately as pickle
                with open(out_file, "wb") as f:
                    pickle.dump((ct_data, mask_data), f)

                print(f"✅ Saved {patient_id} → {out_file}")

            except Exception as e:
                print(f"❌ Error processing {patient_id}: {e}")


if __name__ == "__main__":
    dataset_path = r"/home/hdd4tb_new/users/cherish/Interactive_segmentation/new_model/data/np_test"
    med_dataset = NumpyMedicalDataset(dataset_path, output_dir="processed_np_test_patients")
    med_dataset.process_and_save_all()
