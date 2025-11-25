import numpy as np
import matplotlib.pyplot as plt
import torch

# --------- Config ---------
WTS_PATH = "class_weights.npy"

CLASS_NAMES = [
    "Background",
    "Esophagus",
    "Lung_L",
    "Lung_R",
    "Kidney_L",
    "Kidney_R",
    "Stomach",
    "Liver",
    "Spleen",
    "Pancreas",
    "Intestine"
]

IGNORE_INDEX = 0   # ignore Background


# --------- Load class weights ---------
weights = np.load(WTS_PATH)
print("Loaded class weights:", weights)


# --------- Remove background ---------
weights_no_bg = weights[1:]         # drop index 0
names_no_bg = CLASS_NAMES[1:]       # drop Background label


# --------- Reverse to get frequencies ---------
# original logic: w = 1 / (f + 1e-6), then normalized
inv = 1.0 / (weights_no_bg + 1e-12)
freq = inv / inv.sum()

print("\nEstimated true class frequencies (Background ignored):")
for cls, f in zip(names_no_bg, freq):
    print(f"{cls:12s} : {f:.6f}")


# --------- Plot distribution ---------
plt.figure(figsize=(10, 6))
plt.bar(names_no_bg, freq)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Estimated Class Frequency")
plt.title("Class Frequency Distribution (Ignoring Background)")
plt.tight_layout()
plt.savefig("class_distribution_ignore_bg.png", dpi=200)
plt.close()

print("\n📊 Saved plot as class_distribution_ignore_bg.png")
