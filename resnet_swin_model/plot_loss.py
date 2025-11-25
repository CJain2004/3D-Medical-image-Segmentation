import torch
import os
import matplotlib.pyplot as plt

ckpt_dir = "checkpoints"
epochs = []
losses = []

for file in sorted(os.listdir(ckpt_dir)):
    if file.startswith("epoch_") and file.endswith(".pth"):
        ckpt = torch.load(os.path.join(ckpt_dir, file), map_location="cpu")
        epochs.append(ckpt["epoch"])
        losses.append(ckpt["loss"])

plt.figure(figsize=(8,5))
plt.plot(epochs, losses, marker='o', color='blue')
plt.title("Training Loss Curve (from checkpoints)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_from_ckpts.png")
plt.show()
