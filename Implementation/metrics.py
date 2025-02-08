import torch
import numpy as np

def calculate_metrics(pred, target):
    """
    Computes binary segmentation metrics.
    Assumes that pred is of shape (B, num_classes, H, W)
    and target is of shape (B, H, W) or (B, 1, H, W).
    """
    # Choose the predicted class (assumes background=0, foreground=1).
    pred = torch.argmax(pred, dim=1)
    # Remove singleton channel dimension if necessary.
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    tp = torch.sum((pred == 1) & (target == 1)).float()
    fp = torch.sum((pred == 1) & (target == 0)).float()
    fn = torch.sum((pred == 0) & (target == 1)).float()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    jaccard = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'jaccard': jaccard.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def aggregate_metrics(metrics_list):
    """Aggregate a list of metric dictionaries by taking the mean."""
    return {
        'dice': np.mean([m['dice'] for m in metrics_list]),
        'jaccard': np.mean([m['jaccard'] for m in metrics_list]),
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list])
    }
