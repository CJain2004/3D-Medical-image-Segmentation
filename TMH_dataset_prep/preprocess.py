import os
import pickle
import numpy as np
from scipy.ndimage import zoom

# -----------------
# Config
# -----------------
TARGET_SPACING = (1.0, 1.0, 1.0)   # dummy spacing since np_train has no metadata
TARGET_SHAPE = (256, 256, 256)     # (depth, height, width)
INPUT_DIR = "processed_np_test_patients"       # folder with patient .pkl files (from np_train)
OUTPUT_DIR = "preprocessed_patients_new_test_data"   # folder to save standardized .pkl files
DELETE_ORIGINAL = False

# -----------------
# Utils
# -----------------
def resize_to_shape(image, target_shape, order=1):
    """Resize to fixed shape by zooming."""
    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=order)

def preprocess_and_save_patient(in_file, out_file, target_shape, target_spacing, delete_original=False):
    with open(in_file, "rb") as f:
        ct, mask = pickle.load(f)

    # --- Resize to fixed shape ---
    ct_resized = resize_to_shape(ct, target_shape, order=1)
    mask_resized = resize_to_shape(mask, target_shape, order=0)

    # --- Intensity normalization ---
    ct_resized = (ct_resized - np.mean(ct_resized)) / (np.std(ct_resized) + 1e-8)

    # Save processed in the SAME format as old script
    with open(out_file, "wb") as f:
        pickle.dump((ct_resized.astype(np.float32),
                     mask_resized.astype(np.uint8),
                     target_spacing), f)

    print(f"✅ Saved preprocessed → {out_file}")

    if delete_original:
        try:
            os.remove(in_file)
            print(f"🗑️ Deleted original {in_file}")
        except Exception as e:
            print(f"⚠️ Could not delete {in_file}: {e}")

def preprocess_dataset(input_dir, output_dir, target_shape, target_spacing, delete_original=False):
    os.makedirs(output_dir, exist_ok=True)
    patient_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])

    for fname in patient_files:
        in_file = os.path.join(input_dir, fname)
        out_file = os.path.join(output_dir, fname)

        if os.path.exists(out_file):
            print(f"Skipping {fname} (already preprocessed)")
            if delete_original:
                try:
                    os.remove(in_file)
                    print(f"🗑️ Deleted duplicate original {in_file}")
                except Exception as e:
                    print(f"⚠️ Could not delete {in_file}: {e}")
            continue

        try:
            preprocess_and_save_patient(in_file, out_file, target_shape, target_spacing, delete_original)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

if __name__ == "__main__":
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SHAPE, TARGET_SPACING, delete_original=DELETE_ORIGINAL)
