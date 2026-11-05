import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

# ProteinGPS model from cellmaps_coembedding
from cellmaps_coembedding.runner import ProteinGPSCoEmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True)
    return parser.parse_args()


def union_df(*dfs):
    all_genes = sorted(set().union(*(df.index for df in dfs)))
    reindexed = [df.reindex(all_genes) for df in dfs]
    filled = [r.fillna(0) for r in reindexed]
    counts = sum(r.notna().any(axis=1).astype(int) for r in reindexed)
    summed = sum(filled)
    return summed.div(counts, axis=0)


def write_latent(df, outfile):
    with open(outfile, "w") as f:
        header = [str(ii) for ii in range(df.shape[1])]
        header[0] = ""
        header = "\t".join(header)
        f.write(header + "\n")
        for line in df.reset_index().values:
            current_line = "\t".join([line[0]] + [f"{ii:.10g}" for ii in line[1:]])
            f.write(current_line + "\n")


def main():
    args = parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    outdir = f"../../../result/HEK293T_HCT116/coemb/gps_combine/combine/"
    os.makedirs(outdir, exist_ok=True)
    
    gen = ProteinGPSCoEmbeddingGenerator(
        dimensions=128,
        outdir=outdir,
        embeddings=[
            "../../../result/HEK293T_HCT116/input/BioPlex-HCT116.tsv",
            "../../../result/HEK293T_HCT116/input/SECMX-HCT116.tsv",
            "../../../result/HEK293T_HCT116/input/BioPlex-HEK293T.tsv",
            "../../../result/HEK293T_HCT116/input/SECMX-HEK293T.tsv",
        ],
        n_epochs=200,
        batch_size=64,
        triplet_margin=0.2,
        dropout=0.5,
        l2_norm=True,
        lambda_reconstruction=1.0,
        lambda_l2=0.0,
        lambda_triplet=1.0,
    )

    result = []
    for ii in gen.get_next_embedding():
        result.append(ii)

    # the model is context-agnostic
    # so each context is scored on its own modalities averaged
    for cell in args.cells:
        latents = [
            pd.read_csv(f"{outdir}/proteingps_BioPlex-{cell}_latent.tsv", sep="\t"),
            pd.read_csv(f"{outdir}/proteingps_SECMX-{cell}_latent.tsv", sep="\t"),
        ]
        write_latent(union_df(*latents), f"{outdir}/proteingps_{cell}_latent.tsv")


if __name__ == "__main__":
    main()
