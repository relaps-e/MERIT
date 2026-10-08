"""Ranking objective used by MERIT."""

import torch


def listmle_loss(P_pred, P_true, eps=1e-10):
    """ListMLE normalized by each dataset's observed candidate count."""
    nan_mask = torch.isnan(P_true)
    P_true_filled = P_true.clone()
    P_true_filled[nan_mask] = float("-inf")
    sorted_indices = torch.argsort(P_true_filled, dim=1, descending=True)
    P_pred_sorted = torch.gather(P_pred, dim=1, index=sorted_indices)
    valid_mask_sorted = torch.gather(~nan_mask, dim=1, index=sorted_indices)
    P_pred_masked = P_pred_sorted.clone()
    P_pred_masked[~valid_mask_sorted] = float("-inf")
    P_flipped = torch.flip(P_pred_masked, dims=[1])
    cum_logsumexp_flipped = torch.logcumsumexp(P_flipped, dim=1)
    logsumexp_from_j = torch.flip(cum_logsumexp_flipped, dims=[1])
    loss_per_pos = logsumexp_from_j - P_pred_sorted
    loss_per_pos = torch.nan_to_num(loss_per_pos, nan=0.0, posinf=0.0, neginf=0.0)
    loss_per_pos = loss_per_pos * valid_mask_sorted.float()
    loss_per_sample = loss_per_pos.sum(dim=1)
    valid_counts = valid_mask_sorted.sum(dim=1).float().clamp(min=1.0)
    loss_per_sample = loss_per_sample / valid_counts
    return loss_per_sample.mean()
