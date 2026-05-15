import torch

def dice_coef(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Xử lý trường hợp cả target và pred đều empty (no tumor)
    if target.sum() == 0 and pred.sum() == 0:
        return torch.tensor(1.0, device=pred.device, dtype=torch.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * intersection) / (union + 1e-6)


def iou_coef(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Xử lý trường hợp cả target và pred đều empty (no tumor)
    if target.sum() == 0 and pred.sum() == 0:
        return torch.tensor(1.0, device=pred.device, dtype=torch.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-6)