import math
from collections import defaultdict

import numpy as np
import torch


def nan_mean(values):
    return float(np.mean(values)) if values else float("nan")


def scalar(value):
    return float(value.detach())


class LatentGeometry:
    """
    Track the fraction of latent variance in the top three principal
    components over an epoch.
    """

    def __init__(self, dim, device):
        self.total = torch.zeros(dim, device=device)
        self.gram = torch.zeros(dim, dim, device=device)
        self.count = 0

    def update(self, z):
        z = z.detach()
        self.total += z.sum(dim=0)
        self.gram += z.t() @ z
        self.count += z.shape[0]

    def summary(self):
        if self.count < 2:
            return {}
        mean = self.total / self.count
        cov = self.gram / self.count - torch.outer(mean, mean)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0).flip(0)
        share = eig / eig.sum().clamp_min(1e-12)
        return {
            "top3": float(share[:3].sum()),
        }


def geometry_record(geometry):
    record = {}
    for mod, tracker in geometry.items():
        for key, value in tracker.summary().items():
            record[f"{mod}_latent_{key}"] = value
    return record


def worst_top3(record, modalities):
    values = [
        record[f"{mod}_latent_top3"]
        for mod in modalities
        if f"{mod}_latent_top3" in record
    ]
    return max(values) if values else float("nan")


class CollapseGuard:
    # Always report persistent gate or latent collapse without stopping
    # training. Thresholds are internal and match the original defaults.

    def __init__(self, modalities, watch_gate):
        self.modalities = modalities
        self.watch_gate = watch_gate
        self.strikes = defaultdict(int)
        self._grace = 40
        self._patience = 20
        self._gate_min = 0.005
        self._gate_max = 0.95
        self._top3_max = 0.5

    def check(self, record, epoch, context):
        """
        Print once per sustained collapse episode, allowing training
        to continue. Recovery resets the counter for a later episode.
        """
        if epoch < self._grace:
            self.strikes.clear()
            return
        patience = self._patience

        # gate_open is the fraction of its range the gate is opened to,
        # comparable across gate_angle_max settings
        if self.watch_gate:
            open_frac = record.get("gate_open", float("nan"))
            low = self._gate_min
            high = self._gate_max
            if math.isfinite(open_frac) and low <= open_frac <= high:
                self.strikes["gate"] = 0
            else:
                self.strikes["gate"] += 1
                if self.strikes["gate"] == patience:
                    print(
                        f"WARNING {context} epoch {epoch}: "
                        "the gate has sat at "
                        f"{open_frac:.4f} of its range for {patience} "
                        f"epochs, outside [{low:.3g}, {high:.3g}]. The "
                        "shared/specific split is not being made -- "
                        "raise lambda_gate if it is pinned open, "
                        "lower it or lengthen ramp_epochs if it is "
                        "pinned shut. Training continues"
                    )

        top3 = worst_top3(record, self.modalities)
        top3_max = self._top3_max
        if math.isfinite(top3) and top3 <= top3_max:
            self.strikes["rank"] = 0
        else:
            self.strikes["rank"] += 1
            if self.strikes["rank"] == patience:
                print(
                    f"WARNING {context} epoch {epoch}: three principal "
                    f"components carry {100.0 * top3:.1f}% of the "
                    f"latent variance and have done for {patience} "
                    "epochs. The embedding has collapsed -- raise "
                    "lambda_var and lambda_mean. Training continues"
                )


def summarise(epoch, acc):
    record = {"epoch": epoch}
    for key, values in acc.items():
        record[key] = nan_mean(values)
    return record


def report_epoch(context, epoch, record, params, log_every=25):
    every = log_every
    last = params["n_epochs"] - 1
    if epoch % every != 0 and epoch != last:
        return
    keys = (
        "total",
        "recon",
        "replay",
        "triplet",
        "shared_anchor",
        "mi",
        "gate",
        "gate_open",
    )
    fields = [key for key in keys if key in record]
    body = "  ".join(f"{key} {record[key]:.4f}" for key in fields)
    top3 = [
        value
        for key, value in record.items()
        if key.endswith("_latent_top3")
    ]
    if top3:
        body += f"  top3 {max(top3):.3f}"
    print(f"  {context} epoch {epoch}  {body}")
