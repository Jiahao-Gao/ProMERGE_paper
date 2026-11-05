import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp


class BaseCoEmbed(nn.Module):
    """
    Encoder and decoder per modality.
    Support cross modality reconstruction.
    """
    def __init__(
        self,
        input_dims,
        latent_dim,
        hidden_1,
        hidden_2,
        dropout,
        l2_norm,
    ):
        super().__init__()
        self.l2_norm = l2_norm
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.encoders[name] = make_mlp(
                dim,
                hidden_1,
                hidden_2,
                latent_dim,
                dropout,
            )
            self.decoders[name] = make_mlp(
                latent_dim,
                hidden_2,
                hidden_1,
                dim,
                dropout,
            )

    def encode(self, modality, x):
        raw = self.encoders[modality](x)
        latent = F.normalize(raw, p=2, dim=-1) if self.l2_norm else raw
        return raw, latent

    def decode(self, modality, z):
        return self.decoders[modality](z)
