import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # convert logits → probs

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

bce = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)