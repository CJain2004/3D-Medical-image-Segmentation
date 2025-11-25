import numpy as np
import torch
import random
from scipy.ndimage import label

def ddg_virtual_user(pred_prob, gt_mask, num_classes, click_prob=1.0, max_clicks=3, drop_rate=0.5):
    """
    Simulates user clicks based on disagreement between prediction and ground truth.
    Implements DDG-SIM rules.

    Args:
        pred_prob (torch.Tensor): Prediction probability maps, shape (N, H, W)
        gt_mask (torch.Tensor): Ground truth one-hot mask, shape (N, H, W)
        num_classes (int): Number of classes including background
        click_prob (float): Initial probability of placing a click
        max_clicks (int): Max clicks per connected component
        drop_rate (float): Fraction of clicks randomly dropped

    Returns:
        interaction_mask (torch.Tensor): Binary click mask, shape (N, H, W)
        float: Updated click probability
    """
    device = pred_prob.device

    # Convert tensors to numpy for processing
    pred_prob_np = pred_prob.detach().cpu().numpy()
    gt_mask_np   = gt_mask.detach().cpu().numpy()

    # Convert prob → label maps
    pred_labels = np.argmax(pred_prob_np, axis=0)
    gt_labels   = np.argmax(gt_mask_np, axis=0)

    H, W = pred_labels.shape
    interaction_mask = np.zeros((num_classes, H, W), dtype=np.float32)

    # Rule 3: randomly skip one class
    skip_class = random.randint(0, num_classes - 1)

    for cls in range(num_classes):
        # if cls == skip_class:
        #     continue

        # Disagreement mask
        diff_mask = ((gt_labels == cls) & (pred_labels != cls)) | \
                    ((gt_labels != cls) & (pred_labels == cls))

        if not np.any(diff_mask):
            continue

        # Connected components
        labeled_regions, num_regions = label(diff_mask.astype(np.int32))

        # Sort regions by size
        region_sizes = [(r, np.sum(labeled_regions == r)) for r in range(1, num_regions + 1)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)

        clicks_placed = 0
        for region_label, _ in region_sizes:
            if clicks_placed >= max_clicks:
                break
            if random.random() > click_prob:
                continue  # Rule 2

            coords = np.argwhere(labeled_regions == region_label)
            if coords.shape[0] == 0:
                continue

            cy, cx = coords[len(coords) // 2]  # approximate center
            if cy < H and cx < W:
                interaction_mask[cls, cy, cx] = 1.0
                clicks_placed += 1

    # Rule 4: randomly drop clicks
    mask_indices = np.argwhere(interaction_mask == 1)
    kept = 0
    for c, y, x in mask_indices:
        if random.random() < drop_rate and kept > 0:
            interaction_mask[c, y, x] = 0
        else:
            kept += 1  # keep at least one click if available

    # Update click probability
    # click_prob = max(0.0, click_prob - 1.0 / max_clicks)
    # click_prob *= 0.7   # exponential decay
    click_prob = max(0.2, click_prob - 1.0 / max_clicks)

    # Return torch tensor on same device as input
    return torch.from_numpy(interaction_mask).float().to(device), click_prob