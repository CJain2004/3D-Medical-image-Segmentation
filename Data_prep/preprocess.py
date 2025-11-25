import os
import pickle
import numpy as np
from scipy.ndimage import zoom

# -----------------
# Config
# -----------------
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm
TARGET_SHAPE = (256, 256, 256)     # (depth, height, width)
INPUT_DIR = "processed_patients_ras"       # folder with raw patient .pkl files
OUTPUT_DIR = "preprocessed_patients"       # folder to save processed .pkl files
DELETE_ORIGINAL = True                     # delete input file after processing

# -----------------
# Utils
# -----------------
def resample_image(image, current_spacing, target_spacing, order=1):
    """Resample 3D image to target spacing."""
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    return zoom(image, zoom_factors, order=order)

def resize_to_shape(image, target_shape, order=1):
    """Resize to fixed shape by zooming."""
    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=order)

def preprocess_and_save_patient(in_file, out_file, target_spacing, target_shape, delete_original=False):
    with open(in_file, "rb") as f:
        ct, mask, spacing = pickle.load(f)

    # --- Resample to fixed voxel spacing ---
    ct_resampled = resample_image(ct, spacing, target_spacing, order=1)     # linear for CT
    mask_resampled = resample_image(mask, spacing, target_spacing, order=0) # nearest for mask

    # --- Resize to fixed shape ---
    ct_resized = resize_to_shape(ct_resampled, target_shape, order=1)
    mask_resized = resize_to_shape(mask_resampled, target_shape, order=0)

    # --- Intensity normalization ---
    ct_resized = (ct_resized - np.mean(ct_resized)) / (np.std(ct_resized) + 1e-8)

    # Save processed
    with open(out_file, "wb") as f:
        pickle.dump((ct_resized.astype(np.float32),
                     mask_resized.astype(np.uint8),
                     target_spacing), f)

    print(f"✅ Saved preprocessed → {out_file}")

    # Delete original only after successful save
    if delete_original:
        try:
            os.remove(in_file)
            print(f"🗑️ Deleted original {in_file}")
        except Exception as e:
            print(f"⚠️ Could not delete {in_file}: {e}")


def preprocess_dataset(input_dir, output_dir, target_spacing, target_shape, delete_original=False):
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
            preprocess_and_save_patient(in_file, out_file, target_spacing, target_shape, delete_original)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")


if __name__ == "__main__":
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SPACING, TARGET_SHAPE, delete_original=DELETE_ORIGINAL)
