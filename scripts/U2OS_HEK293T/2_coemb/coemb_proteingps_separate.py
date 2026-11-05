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
    parser.add_argument("--cell", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    outdir = f"../../../result/U2OS_HEK293T/coemb/gps_separate/{args.cell}/"
    os.makedirs(outdir, exist_ok=True)

    gen = ProteinGPSCoEmbeddingGenerator(
        dimensions=128,
        outdir=outdir,
        embeddings=[
            f"../../../result/U2OS_HEK293T/input/APMS-{args.cell}.tsv",
            f"../../../result/U2OS_HEK293T/input/Image-{args.cell}.tsv",
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


if __name__ == "__main__":
    main()
