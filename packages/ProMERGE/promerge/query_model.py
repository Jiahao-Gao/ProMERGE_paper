import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .estimators import (
    CLUB,
    MINE,
)
from .layers import (
    Recalibrate,
    make_mlp,
)


class QueryCoEmbed(nn.Module):
    """
    Per modality trunk + a shared head.
    Then a contrast block (reads the trunk with the base anchor)
    output a specific direction and a scalar gate.
    """

    def __init__(
        self,
        input_dims,
        latent_dim,
        hidden_1,
        hidden_2,
        dropout,
        l2_norm,
        base_decoders,
        gate_angle_max=1.2,
        recalibrate_full=True,
        disentangle_method="CLUB",
    ):
        super().__init__()
        if not 0.0 < gate_angle_max <= 0.5 * math.pi:
            raise ValueError("gate_angle_max must be in (0, pi/2]")
        self.l2_norm = l2_norm
        self.latent_dim = latent_dim
        self.gate_angle_max = gate_angle_max
        self.disentangle_method = disentangle_method
        self.trunks = nn.ModuleDict()
        self.shared_heads = nn.ModuleDict()
        self.contrasts = nn.ModuleDict()
        self.specific_heads = nn.ModuleDict()
        self.gate_heads = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        self.clubs = nn.ModuleDict()
        self.mines = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.trunks[name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
            )
            self.shared_heads[name] = nn.Linear(hidden_2, latent_dim)
            self.contrasts[name] = nn.Sequential(
                nn.Linear(hidden_2 + latent_dim + 1, hidden_2),
                nn.ReLU(),
            )
            self.specific_heads[name] = nn.Linear(hidden_2, latent_dim)
            self.gate_heads[name] = nn.Linear(hidden_2, 1)

            if disentangle_method == "subtract":
                self.fusions[name] = nn.Sequential(
                    nn.Linear(2 * latent_dim, hidden_2),
                    nn.ReLU(),
                    nn.Linear(hidden_2, latent_dim),
                )

            # build decode from base context (or new if no matching one)
            if name in base_decoders:
                self.decoders[name] = copy.deepcopy(base_decoders[name])
                self.adapters[name] = Recalibrate(
                    dim,
                    full=recalibrate_full,
                )
            else:
                self.decoders[name] = make_mlp(
                    latent_dim,
                    hidden_2,
                    hidden_1,
                    dim,
                    dropout,
                )
                self.adapters[name] = nn.Identity()

    def add_club(
        self,
        modality,
        hidden_dim=256,
        logvar_min=-7.0,
        logvar_max=2.0,
    ):
        self.clubs[modality] = CLUB(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    def add_mine(
        self,
        modality,
        hidden_dim=256,
        clamp=5.0,
        ema_decay=0.99,
    ):
        if clamp <= 0:
            raise ValueError(
                "mine_clamp bounds the critic, so it must be positive"
            )
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(
                "mine_ema_decay is a decay rate, so it must be "
                "in (0, 1)"
            )
        self.mines[modality] = MINE(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            clamp=clamp,
            ema_decay=ema_decay,
        )

    def estimators(self):
        if self.disentangle_method == "MINE":
            return self.mines
        return self.clubs

    def fuse(self, modality, shared, residual):
        total = self.fusions[modality](
            torch.cat([shared, residual], dim=-1)
        )
        if self.l2_norm:
            return F.normalize(total, p=2, dim=-1)
        return total

    def replay_latent(self, modality, shared):
        if self.disentangle_method != "subtract":
            return shared
        return self.fuse(modality, shared, torch.zeros_like(shared))

    def encode_subtract(self, modality, x, anchor, has_anchor):
        h = self.trunks[modality](x)
        encoded = F.normalize(
            self.shared_heads[modality](h),
            p=2,
            dim=-1,
        )
        flag = has_anchor.unsqueeze(-1)
        # a protein the base never measured has nothing to subtract
        # from, so its departure is zero
        shared = torch.where(flag, anchor, encoded)
        specific = encoded - shared
        latent = self.fuse(modality, shared, specific)
        gate = specific.norm(p=2, dim=-1, keepdim=True)
        direction = F.normalize(specific, p=2, dim=-1)
        logit = torch.zeros_like(gate)
        return encoded, direction, gate, specific, latent, logit

    def encode(self, modality, x, anchor, has_anchor):
        if self.disentangle_method == "subtract":
            return self.encode_subtract(modality, x, anchor, has_anchor)
        h = self.trunks[modality](x)
        predicted = F.normalize(
            self.shared_heads[modality](h),
            p=2,
            dim=-1,
        )
        flag = has_anchor.unsqueeze(-1)
        shared = torch.where(flag, anchor, predicted)
        contrast = self.contrasts[modality](
            torch.cat([h, shared.detach(), flag.float()], dim=-1)
        )
        direction = F.normalize(
            self.specific_heads[modality](contrast),
            p=2,
            dim=-1,
        )
        logit = self.gate_heads[modality](contrast)
        angle = self.gate_angle_max * torch.sigmoid(logit)
        gate = torch.sin(angle)
        specific = gate * direction
        total = torch.cos(angle) * shared + specific
        latent = total
        if self.l2_norm:
            latent = F.normalize(total, p=2, dim=-1)
        return predicted, direction, gate, specific, latent, logit

    def decode(self, modality, z):
        return self.adapters[modality](self.decoders[modality](z))

    def replay(self, modality, z):
        return self.decoders[modality](z)

    def main_parameters(self):
        return [
            param
            for name, param in self.named_parameters()
            if param.requires_grad
            and not name.startswith(("clubs.", "mines."))
        ]
