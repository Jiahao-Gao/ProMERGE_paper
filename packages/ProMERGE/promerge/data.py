import numpy as np
import pandas as pd
import torch


def dense_batch(matrix, row_of, batch, width, device):
    """
    emb matrix for batch protein, and mask of where is the protein
    """
    rows = row_of[batch]
    mask = rows >= 0
    out = torch.zeros(len(batch), width, device=device)
    out[mask] = matrix[rows[mask]]
    return out, mask


def align_to_base(
    context,
    base_frames,
    query_frames,
    latent_dim,
    align_inputs="procrustes",
    align_min_shared=None,
):
    """
    Rotate each query modality onto its base counterpart. The map
    is orthogonal, so no cosine inside the query context changes.
    """
    if align_inputs not in ("procrustes", "none"):
        raise ValueError("align_inputs must be procrustes or none")
    if align_min_shared is None:
        align_min_shared = latent_dim
    if align_inputs == "none":
        return query_frames
    aligned = dict(query_frames)
    for mod, frame in query_frames.items():
        if mod not in base_frames:
            continue
        base_width = base_frames[mod].shape[1]
        if base_width != frame.shape[1]:
            print(
                f"{context}: modality {mod} has {frame.shape[1]} "
                f"columns against the base's {base_width}, "
                "not aligned"
            )
            continue
        rotated, shared, before, after = procrustes_align(
            query_frame=frame,
            base_frame=base_frames[mod],
            min_shared=align_min_shared,
        )
        if rotated is None:
            print(
                f"{context}: modality {mod} shares only {shared} "
                "proteins with the base, fewer than the "
                f"{align_min_shared} needed to fit a "
                "rotation, not aligned"
            )
            continue
        aligned[mod] = rotated
        print(
            f"{context}: modality {mod} rotated onto the base over "
            f"{shared} shared proteins, mean paired cosine "
            f"{before:.4f} -> {after:.4f}"
        )
    return aligned


def procrustes_align(query_frame, base_frame, min_shared):
    shared = base_frame.index.intersection(query_frame.index)
    if len(shared) < min_shared:
        return None, len(shared), float("nan"), float("nan")
    a = query_frame.loc[shared].to_numpy(dtype="float64")
    b = base_frame.loc[shared].to_numpy(dtype="float64")
    left, _, right = np.linalg.svd(a.T @ b, full_matrices=False)
    rotation = left @ right
    aligned = query_frame.to_numpy(dtype="float64") @ rotation
    before = mean_paired_cosine(a, b)
    after = mean_paired_cosine(a @ rotation, b)
    frame = pd.DataFrame(
        aligned,
        index=query_frame.index,
        columns=query_frame.columns,
    )
    return frame, len(shared), before, after


def mean_paired_cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(axis=1).mean())


class ContextData:
    def __init__(self, context, frames, device):
        self.context = context
        self.device = device
        self.frames = frames
        self.modalities = list(frames.keys())
        self.proteins = sorted(
            set().union(*(set(f.index) for f in frames.values()))
        )
        position = {
            protein: i for i, protein in enumerate(self.proteins)
        }
        self.x = {}
        self.row_of = {}
        self.pool_position = {}
        self.input_dims = {}
        for mod, frame in frames.items():
            self.input_dims[mod] = int(frame.shape[1])
            self.x[mod] = torch.tensor(
                frame.to_numpy(dtype="float32"),
                device=device,
            )
            pool = torch.tensor(
                [position[protein] for protein in frame.index],
                dtype=torch.long,
                device=device,
            )
            row = torch.full(
                (len(self.proteins),),
                -1,
                dtype=torch.long,
                device=device,
            )
            row[pool] = torch.arange(
                len(pool),
                dtype=torch.long,
                device=device,
            )
            self.pool_position[mod] = pool
            self.row_of[mod] = row

    def batch_inputs(self, modality, batch):
        return dense_batch(
            self.x[modality],
            self.row_of[modality],
            batch,
            self.input_dims[modality],
            self.device,
        )

    def sample_negatives(self, modality, positions):
        """
        Triplet negatives drawn from the whole modality
        """
        pool = self.pool_position[modality]
        picks = torch.randint(
            len(pool),
            (len(positions),),
            device=self.device,
        )
        clash = pool[picks] == positions
        picks[clash] = (picks[clash] + 1) % len(pool)
        return self.x[modality][picks], pool[picks]


class AnchorSet:
    """
    The frozen base per-modality latents
    """

    def __init__(self, frames, proteins, latent_dim, device):
        self.device = device
        self.latent_dim = latent_dim
        self.modalities = list(frames.keys())
        position = {protein: i for i, protein in enumerate(proteins)}
        self.s = {}
        self.row_of = {}
        for mod, frame in frames.items():
            kept = [
                protein
                for protein in frame.index
                if protein in position
            ]
            frame = frame.loc[kept]
            self.s[mod] = torch.tensor(
                frame.to_numpy(dtype="float32"),
                device=device,
            )
            row = torch.full(
                (len(proteins),),
                -1,
                dtype=torch.long,
                device=device,
            )
            if len(kept) > 0:
                slot = torch.tensor(
                    [position[protein] for protein in kept],
                    dtype=torch.long,
                    device=device,
                )
                row[slot] = torch.arange(
                    len(kept),
                    dtype=torch.long,
                    device=device,
                )
            self.row_of[mod] = row

    def batch(self, modality, batch):
        return dense_batch(
            self.s[modality],
            self.row_of[modality],
            batch,
            self.latent_dim,
            self.device,
        )


def anchor_batch(anchors, modality, positions, latent_dim, device):
    if modality in anchors.modalities:
        return anchors.batch(modality, positions)
    zeros = torch.zeros(len(positions), latent_dim, device=device)
    absent = torch.zeros(
        len(positions),
        dtype=torch.bool,
        device=device,
    )
    return zeros, absent
