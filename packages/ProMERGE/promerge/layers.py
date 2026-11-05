import torch
import torch.nn as nn


def make_mlp(in_dim, hidden_1, hidden_2, out_dim, dropout):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden_1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Linear(hidden_2, out_dim),
    )


class Recalibrate(nn.Module):
    """
    Additional layer after base-context decoder for query-context.
    Replay should skip this.
    """

    def __init__(self, dim, full=True):
        super().__init__()
        self.full = full
        self.bias = nn.Parameter(torch.zeros(dim))
        if full:
            self.weight = nn.Parameter(torch.eye(dim))
        else:
            self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.full:
            return x @ self.weight.t() + self.bias
        return self.scale * x + self.bias
