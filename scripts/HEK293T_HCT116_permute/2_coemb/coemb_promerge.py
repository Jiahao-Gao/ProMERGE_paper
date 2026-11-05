import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

# ProMERGE model
sys.path.insert(0, "../../../packages/ProMERGE")
from runner import ProteinCoEmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_base", required=True)
    parser.add_argument("--cell_query", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--disentangle_method", default="CLUB",
        choices=["CLUB", "MINE", "subtract"]
    )
    parser.add_argument("--lambda_reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda_replay", type=float, default=5.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--lambda_mean", type=float, default=1.0)
    parser.add_argument("--lambda_triplet", type=float, default=0.2)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument(
        "--align_inputs", default="procrustes",
        choices=["procrustes", "none"]
    )
    parser.add_argument("--save_update_epochs", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    run_stem = {
        "CLUB": "promergeclub",
        "MINE": "promergemine",
        "subtract": "promergesubtract",
    }
    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"{run_stem[args.disentangle_method]}_"
            f"{args.cell_base}anchor"
        )
    outdir = f"../../../result/HEK293T_HCT116_permute/coemb/{run_name}"
    os.makedirs(outdir, exist_ok=True)

    gen = ProteinCoEmbeddingGenerator(
        outdir=outdir,
        embeddings=[
            f"../../../result/HEK293T_HCT116_permute/input/BioPlex-{args.cell_base}.tsv",
            f"../../../result/HEK293T_HCT116_permute/input/SECMX-{args.cell_base}.tsv",
            f"../../../result/HEK293T_HCT116_permute/input/BioPlex-{args.cell_query}.tsv",
            f"../../../result/HEK293T_HCT116_permute/input/SECMX-{args.cell_query}.tsv",
        ],
        contexts=[args.cell_base, args.cell_query],
        modalities=["BioPlex", "SECMX"],
        disentangle_method=args.disentangle_method,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_replay=args.lambda_replay,
        lambda_triplet=args.lambda_triplet,
        lambda_mi=args.lambda_mi,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        align_inputs=args.align_inputs,
        save_update_epochs=args.save_update_epochs,
        seed=1,
    )

    result = []
    for row in gen.get_next_embedding():
        result.append(row)


if __name__ == "__main__":
    main()
