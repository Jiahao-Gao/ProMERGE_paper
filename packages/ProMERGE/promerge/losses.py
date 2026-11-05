import math

import torch
import torch.nn.functional as F


def cosine_distance(a, b):
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def triplet_loss(anchor, positive, negative, margin, weight=None):
    a = F.normalize(anchor, p=2, dim=-1)
    p = F.normalize(positive, p=2, dim=-1)
    n = F.normalize(negative, p=2, dim=-1)
    per_sample = F.relu(margin + (a * n).sum(-1) - (a * p).sum(-1))
    if weight is None:
        return per_sample.mean()
    return (weight * per_sample).sum() / (weight.sum() + 1e-6)


def variance_target(latent_dim, l2_norm):
    return 1.0 / math.sqrt(latent_dim) if l2_norm else 1.0


def variance_penalty(z, target):
    return F.relu(target - z.std(dim=0)).mean()


def mean_offset_penalty(z):
    floor = 1.0 / math.sqrt(z.shape[0])
    return F.relu(z.mean(dim=0).norm(p=2) - floor)


def gate_penalty(logit):
    """
    The sparsity prior on the gate, read off the logit. Its gradient is
    sigmoid(logit), so it keeps its grip where the gate is wide open
    without driving the gate to a hard zero.
    """
    return F.softplus(logit).mean()


def mean_terms(terms, device):
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).mean()
