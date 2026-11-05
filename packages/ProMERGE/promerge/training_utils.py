import random

import numpy as np
import torch


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ramp_weight(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(ramp_epochs))


def epoch_batches(n_proteins, batch_size, device):
    """
    Permutes the protein positions
    """
    order = torch.randperm(n_proteins, device=device)
    for start in range(0, n_proteins, batch_size):
        yield order[start:start + batch_size]


def snapshot_due(params, epoch):
    """
    Save every save_epoch epoch (including epoch0)
    """
    if not params["save_update_epochs"]:
        return False
    return epoch % params["save_epoch"] == 0
