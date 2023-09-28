import torch
import numpy as np
import torch.nn.functional as F

def zeroes(x, aug_percent, device, mix=0., scale=0.):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly sets some percentage to zero
    """
    aug_x = F.dropout(x.detach().clone(), aug_percent) * (1. - aug_percent)
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * scale
    return noise_aug_x    


def qgswap(x, aug_percent, device, mix=0., scale=0.):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly swaps some percentage in-qg
    """
    qg_dim = x.shape[1]
    corrupted_indices_cont = torch.rand(x.shape).to(device)

    corrupted_indices_indicator = (corrupted_indices_cont < aug_percent).to(device)
    dim0_target, dim1_target, dim2_target = torch.where(corrupted_indices_indicator)
    dim0_target, dim1_target, dim2_target = dim0_target.to(device), dim1_target.to(device), dim2_target.to(device)
    dim1_source = torch.randint(0, qg_dim, size=dim1_target.shape).to(device)

    aug_x = x.detach().clone().to(device)
    aug_x[dim0_target, dim1_target, dim2_target] = x[dim0_target, dim1_source, dim2_target].detach().clone().to(device)

    return aug_x

def gaussian(x, aug_percent, device):
    aug_x = x.detach().clone().to(device)
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * aug_percent
    return noise_aug_x

def qg_and_zero(x, aug_percent, device, mix=0., scale=0.):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly swaps some percentage in-qg
    """
    qg_dim = x.shape[1]
    corrupted_indices_cont = torch.rand(x.shape).to(device)

    corrupted_indices_indicator = (corrupted_indices_cont < aug_percent).to(device)
    dim0_target, dim1_target, dim2_target = torch.where(corrupted_indices_indicator)
    dim0_target, dim1_target, dim2_target = dim0_target.to(device), dim1_target.to(device), dim2_target.to(device)
    dim1_source = torch.randint(0, qg_dim, size=dim1_target.shape).to(device)

    aug_x = x.detach().clone().to(device)
    candidate_replacements = x[dim0_target, dim1_source, dim2_target].to(device)
    random_zero_percent = mix
    candidate_replacements_zero = F.dropout(candidate_replacements, random_zero_percent) * (1. - random_zero_percent)
    aug_x[dim0_target, dim1_target, dim2_target] = candidate_replacements_zero
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * scale
    return noise_aug_x    
