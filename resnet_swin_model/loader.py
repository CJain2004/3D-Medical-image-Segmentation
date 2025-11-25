import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class MedicalSegmentationDataset(Dataset):
    def __init__(self, pkl_dir, classes, sim_depth=3, transform=None, load_sim=False):
        """
        Args:
            pkl_dir (str): Path to folder containing patient-level pickle files
            classes (list): List of class names (including background)
            sim_depth (int): Number of sequential memory states D
            transform (callable): Optional transform for data augmentation
            load_sim (bool): Whether to initialize with empty SIM tensor
        """
        self.pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
        assert len(self.pkl_files) > 0, f"No pickle files found in {pkl_dir}"

        self.classes = classes
        self.num_classes = len(classes)
        self.sim_depth = sim_depth
        self.transform = transform
        self.load_sim = load_sim

        # Build slice index: (file_path, patient_id, slice_idx)
        self.slice_index = []
        total_slices = 0

        print(f"[Dataset] Scanning {len(self.pkl_files)} pickle files in {pkl_dir}")
        for f_i, fpath in enumerate(self.pkl_files, 1):
            pid = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath, "rb") as f:
                ct, mask, spacing = pickle.load(f)

            num_slices = ct.shape[0]
            total_slices += num_slices

            for idx in range(num_slices):
                # if np.any(mask[idx] > 0):  # keep only slices with at least one foreground pixel
                self.slice_index.append((fpath, pid, idx))
            # for idx in range(num_slices):
            #     if np.any(mask[idx] > 0):
            #         # Always keep slices with at least one foreground pixel
            #         self.slice_index.append((fpath, pid, idx))
            #     else:
            #         # Keep some empty slices (e.g. 10% chance)
            #         if np.random.rand() < 0.3:
            #             self.slice_index.append((fpath, pid, idx))

            # 🔹 Print progress as we go
            print(f"  [{f_i}/{len(self.pkl_files)}] {pid}: {num_slices} slices "
                  f"(running total: {total_slices})")

        print(f"[Dataset] Initialization complete. Total slices: {total_slices}")

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        fpath, pid, slice_idx = self.slice_index[idx]

        # Load patient data only when needed
        with open(fpath, "rb") as f:
            ct, mask, spacing = pickle.load(f)

        image = ct[slice_idx]   # (H, W)
        mask = mask[slice_idx]  # (H, W)

        # One-hot encode mask
        mask_oh = np.zeros((self.num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for c in range(self.num_classes):
            mask_oh[c] = (mask == c).astype(np.float32)

        # Prepare SIM tensor
        if self.load_sim:
            sim_channels = 2 * self.sim_depth * self.num_classes
            sim_tensor = np.zeros((sim_channels, image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            sim_tensor = None

        # Apply transform if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask_oh)
            image = augmented["image"]
            mask_oh = augmented["mask"]

        # Convert to torch tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_oh).float()              # (N, H, W)
        if sim_tensor is not None:
            sim_tensor = torch.from_numpy(sim_tensor).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sim": sim_tensor,
            "patient_id": pid,
            "slice_idx": slice_idx,
            "spacing": spacing
        }
