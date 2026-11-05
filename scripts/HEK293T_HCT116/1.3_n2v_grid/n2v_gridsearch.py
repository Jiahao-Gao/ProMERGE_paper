import argparse
import itertools
import os
import random
import time
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from cellmaps_ppi_embedding.runner import Node2VecEmbeddingGenerator

SEARCH_GRID = [
    {
        "p": p,
        "q": q,
        "dimensions": dimensions,
        "walk_length": walk_length,
        "num_walks": num_walks,
    }
    for p, q, dimensions, (walk_length, num_walks) in (
        itertools.product(
            [0.25, 0.5, 1, 2, 4],
            [0.25, 0.5, 1, 2, 4],
            [128, 256],
            [(80, 10), (160, 20), (320, 40)],
        )
    )
]


def load_network(infile):
    df = pd.read_csv(infile, sep="\t")
    g = nx.Graph()
    g.add_edges_from(df[['GeneA', 'GeneB']].values.tolist())
    return g


def run_n2v(g, config, seed):
    random.seed(seed)
    np.random.seed(seed)

    gen = Node2VecEmbeddingGenerator(
        nx_network=g,
        dimensions=config["dimensions"],
        p=config["p"],
        q=config["q"],
        walk_length=config["walk_length"],
        num_walks=config["num_walks"],
        workers=1
    )
    return pd.DataFrame(list(gen.get_next_embedding())).set_index(0)


def pair_scores(X, pairs):
    """
    Cosine similarity of the given index pairs, from L2-normalised rows.
    """
    return (X[pairs[:, 0]] * X[pairs[:, 1]]).sum(axis=1)


def precision_at_k(X, edges, k, chunk=2048):
    """
    Fraction of the k most similar gene pairs in the whole matrix that are real edges.
    """
    n = X.shape[0]
    top_scores = np.empty(0, dtype=np.float32)
    top_pairs = np.empty((0, 2), dtype=np.int32)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = X[start:stop] @ X.T
        # keep each unordered pair once by dropping everything at or below the diagonal
        block[np.arange(n)[None, :] <= (start + np.arange(stop - start))[:, None]] = -np.inf

        flat = block.ravel()
        take = min(k, flat.size)
        picked = np.argpartition(-flat, take - 1)[:take]
        picked = picked[np.isfinite(flat[picked])]

        top_scores = np.concatenate([top_scores, flat[picked]])
        top_pairs = np.concatenate([
            top_pairs,
            np.stack([start + picked // n, picked % n], axis=1).astype(np.int32),
        ])
        if len(top_scores) > k:
            keep = np.argpartition(-top_scores, k - 1)[:k]
            top_scores, top_pairs = top_scores[keep], top_pairs[keep]

    return sum(1 for pair in map(tuple, top_pairs) if pair in edges) / k


def evaluate(emb, g, seed, neg_ratio=10):
    """
    AUROC of edges against sampled non-edges, and precision at the number of edges.
    """
    genes = list(emb.index)
    index = {gene: i for i, gene in enumerate(genes)}
    X = emb.values.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    pos = np.sort(np.array([[index[a], index[b]] for a, b in g.edges()]), axis=1)
    edges = set(map(tuple, pos))

    # uniform non-edges
    rng = np.random.default_rng(seed)
    n_neg = neg_ratio * len(pos)
    draw = rng.integers(0, len(genes), size=(int(n_neg * 1.2), 2))
    draw = np.sort(draw[draw[:, 0] != draw[:, 1]], axis=1)
    neg = np.array([pair for pair in map(tuple, draw) if pair not in edges][:n_neg])

    scores = np.concatenate([pair_scores(X, pos), pair_scores(X, neg)])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    return roc_auc_score(labels, scores), precision_at_k(X, edges, len(pos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config_id", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_ratio", type=int, default=10)
    args = parser.parse_args()

    config = SEARCH_GRID[args.config_id]
    network = os.path.basename(args.infile).rsplit(".", 1)[0]
    g = load_network(args.infile)
    emb = run_n2v(g, config, args.seed)
    auroc, precision = evaluate(emb, g, args.seed, neg_ratio=args.neg_ratio)

    row = dict(config)
    row.update({
        "network": network,
        "config_id": args.config_id,
        "seed": args.seed,
        "n_gene": g.number_of_nodes(),
        "n_edge": g.number_of_edges(),
        "auroc": auroc,
        "precision_at_E": precision,
    })
    row = pd.DataFrame([row])

    os.makedirs(args.outdir, exist_ok=True)
    row.to_csv(
        f"{args.outdir}/{network}__cfg{args.config_id:03d}__seed{args.seed}.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
