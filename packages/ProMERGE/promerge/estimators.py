import torch
import torch.nn as nn


LOG2PI = 1.8378770664093453


class CLUB(nn.Module):
    def __init__(self, latent_dim, hidden_dim, logvar_min, logvar_max):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_center = 0.5 * (logvar_max + logvar_min)
        self.logvar_span = 0.5 * (logvar_max - logvar_min)

    def forward(self, z):
        h = self.body(z)
        mu = self.mu_head(h)
        raw = self.logvar_head(h) / self.logvar_span
        logvar = self.logvar_center + self.logvar_span * torch.tanh(raw) # tanh bounds the variance
        return mu, logvar

    def loglik(self, z, s):
        mu, logvar = self(z)
        density = ((s - mu) ** 2) / logvar.exp() + logvar + LOG2PI
        return -0.5 * density.sum(-1)

    def bound(self, z, s):
        mu, logvar = self(z)
        var = logvar.exp()
        matched = -0.5 * (
            ((s - mu) ** 2) / var + logvar + LOG2PI
        ).sum(-1)
        gap = s.unsqueeze(0) - mu.unsqueeze(1)
        mismatched = -0.5 * (
            (gap ** 2) / var.unsqueeze(1)
            + (logvar + LOG2PI).unsqueeze(1)
        ).sum(-1)
        return matched.mean() - mismatched.mean(), matched, var


class MINE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, clamp, ema_decay):
        super().__init__()
        self.clamp = clamp
        self.ema_decay = ema_decay
        self.body = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("ema", torch.ones(()))

    def forward(self, z, s):
        h = self.body(torch.cat([z, s], dim=-1))
        raw = self.score_head(h).squeeze(-1)
        return self.clamp * torch.tanh(raw / self.clamp), raw

    def marginal_pairs(self, s):
        n = s.shape[0]
        picks = torch.randperm(n, device=s.device)
        clash = picks == torch.arange(n, device=s.device)
        picks[clash] = (picks[clash] + 1) % n
        return s[picks]

    def estimate(self, z, s, update_ema):
        joint, _ = self(z, s)
        marginal, _ = self(z, self.marginal_pairs(s))
        mean_exp = marginal.exp().mean()
        if update_ema:
            with torch.no_grad():
                self.ema.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * mean_exp.detach()
                )
        ratio = mean_exp / self.ema.detach().clamp_min(1e-12)
        value = (
            joint.mean()
            - ratio
            + ratio.detach()
            - mean_exp.detach().clamp_min(1e-12).log()
        )
        return value
