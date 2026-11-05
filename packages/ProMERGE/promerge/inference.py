from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .data import anchor_batch


@torch.no_grad()
def encode_base(model, data, chunk=2048):
    model.eval()
    latents = {}
    for mod in data.modalities:
        parts = []
        matrix = data.x[mod]
        for start in range(0, len(matrix), chunk):
            _, latent = model.encode(mod, matrix[start:start + chunk])
            parts.append(latent.cpu().numpy())
        latents[mod] = pd.DataFrame(
            np.concatenate(parts),
            index=list(data.frames[mod].index),
        )
    return latents


@torch.no_grad()
def encode_query(model, data, anchors, latent_dim, device, chunk=2048):
    model.eval()
    latents = {}
    specifics = {}
    for mod in data.modalities:
        latent_parts = []
        specific_parts = []
        matrix = data.x[mod]
        pool = data.pool_position[mod]
        for start in range(0, len(matrix), chunk):
            positions = pool[start:start + chunk]
            anchor, has_anchor = anchor_batch(
                anchors,
                mod,
                positions,
                latent_dim,
                device,
            )
            encoded = model.encode(
                mod,
                matrix[start:start + chunk],
                anchor,
                has_anchor,
            )
            latent_parts.append(encoded[4].cpu().numpy())
            specific_parts.append(encoded[3].cpu().numpy())
        index = list(data.frames[mod].index)
        latents[mod] = pd.DataFrame(
            np.concatenate(latent_parts),
            index=index,
        )
        specifics[mod] = pd.DataFrame(
            np.concatenate(specific_parts),
            index=index,
        )
    return latents, specifics


@torch.no_grad()
def reconstruct_base(model, data, chunk=2048):
    model.eval()
    parts = defaultdict(list)
    for source in data.modalities:
        matrix = data.x[source]
        for start in range(0, len(matrix), chunk):
            _, latent = model.encode(
                source,
                matrix[start:start + chunk],
            )
            for target in data.modalities:
                parts[(source, target)].append(
                    model.decode(target, latent).cpu().numpy()
                )
    return {
        key: pd.DataFrame(
            np.concatenate(value),
            index=list(data.frames[key[0]].index),
        )
        for key, value in parts.items()
    }


@torch.no_grad()
def reconstruct_query(
    model,
    data,
    anchors,
    latent_dim,
    device,
    chunk=2048,
):
    model.eval()
    parts = defaultdict(list)
    for source in data.modalities:
        matrix = data.x[source]
        pool = data.pool_position[source]
        for start in range(0, len(matrix), chunk):
            positions = pool[start:start + chunk]
            anchor, has_anchor = anchor_batch(
                anchors,
                source,
                positions,
                latent_dim,
                device,
            )
            latent = model.encode(
                source,
                matrix[start:start + chunk],
                anchor,
                has_anchor,
            )[4]
            for target in data.modalities:
                parts[(source, target)].append(
                    model.decode(target, latent).cpu().numpy()
                )
    return {
        key: pd.DataFrame(
            np.concatenate(value),
            index=list(data.frames[key[0]].index),
        )
        for key, value in parts.items()
    }
