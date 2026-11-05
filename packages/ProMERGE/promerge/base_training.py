import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn

from .base_model import BaseCoEmbed
from .diagnostics import (
    CollapseGuard,
    LatentGeometry,
    geometry_record,
    report_epoch,
    scalar,
    summarise,
)
from .losses import (
    cosine_distance,
    mean_offset_penalty,
    mean_terms,
    triplet_loss,
    variance_penalty,
    variance_target,
)
from .training_utils import (
    epoch_batches,
    snapshot_due,
)


def make_base(data, params, device):
    return BaseCoEmbed(
        input_dims=data.input_dims,
        latent_dim=params["latent_dim"],
        hidden_1=params["hidden_size_1"],
        hidden_2=params["hidden_size_2"],
        dropout=params["dropout"],
        l2_norm=params["l2_norm"],
    ).to(device)


def load_base(folder, context, data, params, device):
    """
    Rebuild stage 1 from {context}_model.pth.
    The anchors are re-encoded from these weights.
    """
    model = make_base(data, params, device)
    path = os.path.join(folder, context, f"{context}_model.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def train_base(data, params, device, snapshot=None):
    """
    Stage 1: per-modality autoencoders on the base context.
    Triplet loss align modalities.
    All input modalities reconstruct all output modalities (include cross-modalities)
    """
    model = make_base(data, params, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learn_rate"],
    )
    latent_dim = params["latent_dim"]
    var_target = variance_target(latent_dim, params["l2_norm"])
    guard = CollapseGuard(data.modalities, watch_gate=False)
    records = []

    for epoch in range(params["n_epochs"]):
        model.train()
        acc = defaultdict(list)
        geometry = {
            mod: LatentGeometry(latent_dim, device)
            for mod in data.modalities
        }
        batches = epoch_batches(
            len(data.proteins),
            params["batch_size"],
            device,
        )
        for batch in batches:
            x = {}
            mask = {}
            latent = {}
            for mod in data.modalities:
                x[mod], mask[mod] = data.batch_inputs(mod, batch)
                _, latent[mod] = model.encode(mod, x[mod])

            # reconstruction
            recon = []
            for source in data.modalities:
                for target in data.modalities:
                    both = mask[source] & mask[target]
                    if not bool(both.any()):
                        continue
                    out = model.decode(target, latent[source])
                    term = cosine_distance(out[both], x[target][both])
                    term = term.mean()
                    recon.append(term)
                    acc[f"{target}_recon"].append(scalar(term))

            # triplet align the same protein across modalities
            triplet = []
            for first in data.modalities:
                for second in data.modalities:
                    if first == second:
                        continue
                    both = mask[first] & mask[second]
                    if not bool(both.any()):
                        continue
                    negatives, _ = data.sample_negatives(
                        second,
                        batch[both],
                    )
                    _, neg_latent = model.encode(second, negatives)
                    triplet.append(
                        triplet_loss(
                            latent[first][both],
                            latent[second][both],
                            neg_latent,
                            params["triplet_margin"],
                        )
                    )

            # variance and mean penalty
            variance = []
            offset = []
            for mod in data.modalities:
                if int(mask[mod].sum()) < 2:
                    continue
                chosen = latent[mod][mask[mod]]
                variance.append(variance_penalty(chosen, var_target))
                offset.append(mean_offset_penalty(chosen))
                geometry[mod].update(chosen)

            # total loss
            recon_term = mean_terms(recon, device)
            triplet_term = mean_terms(triplet, device)
            variance_term = mean_terms(variance, device)
            offset_term = mean_terms(offset, device)
            total = (
                params["lambda_reconstruction"] * recon_term
                + params["lambda_triplet"] * triplet_term
                + params["lambda_var"] * variance_term
                + params["lambda_mean"] * offset_term
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                params["grad_clip"],
            )
            optimizer.step()

            acc["total"].append(scalar(total))
            acc["recon"].append(scalar(recon_term))
            acc["triplet"].append(scalar(triplet_term))
            acc["var"].append(scalar(variance_term))
            acc["mean_offset"].append(scalar(offset_term))

        record = summarise(epoch, acc)
        record.update(geometry_record(geometry))
        records.append(record)
        report_epoch(data.context, epoch, record, params)
        # Write scheduled snapshots before reporting collapse warnings.
        if snapshot is not None and snapshot_due(params, epoch):
            snapshot(model, epoch)
        guard.check(record, epoch, data.context)

    return model, pd.DataFrame(records)
