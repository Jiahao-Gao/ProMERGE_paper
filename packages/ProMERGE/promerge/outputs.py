import os

import numpy as np
import pandas as pd
import torch


def average_frames(frames, proteins, normalize):
    """
    Average per-modality dataframeså
    """
    position = {protein: i for i, protein in enumerate(proteins)}
    width = next(iter(frames.values())).shape[1]
    total = np.zeros((len(proteins), width), dtype="float64")
    count = np.zeros((len(proteins), 1), dtype="float64")
    for frame in frames.values():
        rows = np.array(
            [position[protein] for protein in frame.index],
            dtype="int64",
        )
        np.add.at(total, rows, frame.to_numpy(dtype="float64"))
        np.add.at(count, rows, 1.0)
    total = total / np.maximum(count, 1.0)
    if normalize:
        norm = np.linalg.norm(total, axis=1, keepdims=True)
        total = total / np.maximum(norm, 1e-12)
    return pd.DataFrame(total, index=proteins)


def specific_score_frame(frames, proteins):
    """
    specific_score is the mean of ||specific|| over the modalities
    """
    position = {protein: i for i, protein in enumerate(proteins)}
    total = np.zeros(len(proteins), dtype="float64")
    count = np.zeros(len(proteins), dtype="float64")
    columns = {}
    for mod, frame in frames.items():
        rows = np.array(
            [position[protein] for protein in frame.index],
            dtype="int64",
        )
        norms = np.linalg.norm(frame.to_numpy(dtype="float64"), axis=1)
        column = np.full(len(proteins), np.nan, dtype="float64")
        column[rows] = norms
        columns[mod] = column
        np.add.at(total, rows, norms)
        np.add.at(count, rows, 1.0)
    frame = pd.DataFrame(columns, index=proteins)
    frame["n_modality"] = count.astype("int64")
    frame["specific_score"] = np.where(
        count > 0,
        total / np.maximum(count, 1.0),
        np.nan,
    )
    return frame


def write_frame(frame, path):
    frame = frame.copy()
    frame.index.name = None
    frame.to_csv(path, sep="\t")


def write_latents(folder, stem, latents, proteins, l2_norm):
    for mod, frame in latents.items():
        name = f"{stem}_{mod}_latent.tsv"
        write_frame(frame, os.path.join(folder, name))
    combined = average_frames(latents, proteins, l2_norm)
    write_frame(combined, os.path.join(folder, f"{stem}_latent.tsv"))
    return combined


def write_reconstructions(folder, stem, reconstructions):
    for (source, target), frame in reconstructions.items():
        name = f"{stem}_{source}___{target}_reconstructed.tsv"
        write_frame(frame, os.path.join(folder, name))


def write_snapshot(
    outdir,
    data,
    model,
    epoch,
    latents,
    specifics,
    l2_norm,
):
    folder = os.path.join(outdir, data.context)
    os.makedirs(folder, exist_ok=True)
    stem = f"{data.context}_epoch{epoch}"
    write_latents(folder, stem, latents, data.proteins, l2_norm)
    if specifics is not None:
        effect = average_frames(
            specifics,
            data.proteins,
            normalize=False,
        )
        write_frame(
            effect,
            os.path.join(folder, f"{stem}_specific.tsv"),
        )
    path = os.path.join(folder, f"{stem}_model.pth")
    torch.save(model.state_dict(), path)


def support_frame(data, base_proteins):
    columns = {}
    for mod in data.modalities:
        measured = set(data.frames[mod].index)
        columns[mod] = [
            int(protein in measured)
            for protein in data.proteins
        ]
    frame = pd.DataFrame(columns, index=data.proteins)
    if base_proteins is not None:
        frame["base"] = [
            int(protein in base_proteins)
            for protein in data.proteins
        ]
    return frame


def write_context(
    outdir,
    data,
    model,
    log,
    latents,
    reconstructions,
    specifics,
    base_proteins,
    l2_norm,
):
    context = data.context
    folder = os.path.join(outdir, context)
    os.makedirs(folder, exist_ok=True)

    combined = write_latents(
        folder,
        context,
        latents,
        data.proteins,
        l2_norm,
    )
    write_reconstructions(folder, context, reconstructions)

    if specifics is not None:
        effect = average_frames(
            specifics,
            data.proteins,
            normalize=False,
        )
        write_frame(
            effect,
            os.path.join(folder, f"{context}_specific.tsv"),
        )
        score = specific_score_frame(specifics, data.proteins)
        write_frame(
            score,
            os.path.join(folder, f"{context}_specific_score.tsv"),
        )

    write_frame(
        support_frame(data, base_proteins),
        os.path.join(folder, f"{context}_support.tsv"),
    )
    log.to_csv(
        os.path.join(folder, f"{context}_log.tsv"),
        sep="\t",
        index=False,
    )
    path = os.path.join(folder, f"{context}_model.pth")
    torch.save(model.state_dict(), path)
    return combined
