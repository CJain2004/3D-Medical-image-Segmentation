# show_class_wts.py
# -----------------------------------
# Load and display class weights from .npy file
# -----------------------------------

import numpy as np
import os

# Path to your saved .npy file
WTS_PATH = "class_weights.npy"   # change path if it's elsewhere

if not os.path.exists(WTS_PATH):
    raise FileNotFoundError(f"❌ File not found: {WTS_PATH}")

# Load weights
class_wts = np.load(WTS_PATH)

# Create class labels
classes = [f"class_{i}" for i in range(len(class_wts))]

# Print results
print("\n📊 Class Weights Representation:")
for cls, wt in zip(classes, class_wts):
    print(f"  {cls:12s}: {wt:.6f}")
