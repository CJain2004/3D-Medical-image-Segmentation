import torch
import torch.nn as nn
import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps

#     def forward(self, logits, targets):
#         """
#         logits: (B, C, H, W)
#         targets: (B, H, W) class indices
#         """
#         num_classes = logits.shape[1]
#         targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
#         probs = torch.softmax(logits, dim=1)

#         dims = (0, 2, 3)
#         intersection = torch.sum(probs * targets_onehot, dims)
#         cardinality = torch.sum(probs + targets_onehot, dims)
#         dice = (2. * intersection + self.eps) / (cardinality + self.eps)
#         return 1. - dice.mean()


# class TverskyLoss(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, eps=1e-6):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.eps = eps

#     def forward(self, logits, targets):
#         num_classes = logits.shape[1]
#         targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
#         probs = torch.softmax(logits, dim=1)

#         dims = (0, 2, 3)
#         TP = torch.sum(probs * targets_onehot, dims)
#         FP = torch.sum(probs * (1 - targets_onehot), dims)
#         FN = torch.sum((1 - probs) * targets_onehot, dims)

#         tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
#         return 1. - tversky.mean()


# class ComboLoss(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, dice_weight=0.7, tversky_weight=0.3):
#         super().__init__()
#         self.dice = DiceLoss()
#         self.tversky = TverskyLoss(alpha=alpha, beta=beta)
#         self.dw = dice_weight
#         self.tw = tversky_weight

#     def forward(self, logits, targets):
#         dice = self.dice(logits, targets)
#         tversky = self.tversky(logits, targets)
#         return self.dw * dice + self.tw * tversky

class WeightedCELoss(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        if weights is None:
            weights = torch.ones(num_classes)
            weights[0] = 0.5   # suppress background
        self.register_buffer("weights", weights)  # ✅ keep it with model

    def forward(self, logits, targets):
        # ensure weights are on same device as logits
        ce = nn.CrossEntropyLoss(weight=self.weights.to(logits.device))
        return ce(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1. - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, weights=None):
        super().__init__()
        self.ce_loss = WeightedCELoss(num_classes, weights)
        self.dice_loss = DiceLoss()
        self.cw = ce_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.cw * ce + self.dw * dice
