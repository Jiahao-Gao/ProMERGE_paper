import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

# the autoencoder fork of cellmaps_coembedding
sys.path.insert(0, "../../../packages/cellmaps_coembedding_1102243_AE")
from cellmaps_coembedding.runner import ProteinGPSCoEmbeddingGenerator

sys.path.insert(0, "../../../packages/ProMERGE_utils")
from promerge_utils import concat_emb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    outdir = f"../../../result/HEK293T_HCT116_permute/coemb/AE/{args.cell}/"
    os.makedirs(outdir, exist_ok=True)
    
    gen = ProteinGPSCoEmbeddingGenerator(
        dimensions=128,
        outdir=outdir,
        embeddings=[
            f"../../../result/HEK293T_HCT116_permute/input/BioPlex-{args.cell}.tsv",
            f"../../../result/HEK293T_HCT116_permute/input/SECMX-{args.cell}.tsv",
        ],
        n_epochs=200,
        batch_size=64,
        triplet_margin=0.2,
        dropout=0.5,
        l2_norm=True,
        lambda_reconstruction=1.0,
        lambda_l2=0.0,
        lambda_triplet=0.0, # keep zero for AE
    )

    result = []
    for ii in gen.get_next_embedding():
        result.append(ii)

    # the autoencoder has no shared space, so the per-modality latents are concatenated
    latents = [
        pd.read_csv(
            f"{outdir}/proteingps_BioPlex-{args.cell}_latent.tsv",
            sep="\t", index_col=0
        ),
        pd.read_csv(
            f"{outdir}/proteingps_SECMX-{args.cell}_latent.tsv",
            sep="\t", index_col=0
        ),
    ]
    df_concat = concat_emb(latents)
    df_concat.to_csv(
        f"{outdir}/Concat-{args.cell}_latent.tsv",
        sep="\t", header=True, index=True
    )


if __name__ == "__main__":
    main()
