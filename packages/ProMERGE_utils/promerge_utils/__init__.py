import os
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .map import (
    check_id_mapping_results_ready,
    get_id_mapping_results_link,
    get_id_mapping_results_search,
    submit_id_mapping,
)


def ppi_from_raw_ppi(infile, outfile):
    """
    Symmetrize a GeneA/GeneB/Score table into a ranked pair list.
    Keep GeneA < GeneB orientation, sorted by score
    descending, and written as geneA__geneB and score with no header.
    """
    df = pd.read_csv(infile, sep="\t")
    flipped = df[["GeneB", "GeneA", "Score"]].values
    df = pd.DataFrame(np.concatenate([df.values, flipped]))
    df = df[df[0] < df[1]].copy()
    df.columns = ["GeneA", "GeneB", "Score"]
    df = df.sort_values("Score", ascending=False)
    df["ppi"] = df["GeneA"] + "__" + df["GeneB"]
    df[["ppi", "Score"]].to_csv(
        outfile,
        sep="\t",
        index=False,
        header=False,
    )


def _as_list(value):
    """
    None to [], a scalar to a one-element list, anything else to a list.
    """
    if value is None:
        return []
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        return [value]
    return list(value)


def ppi_from_emb(
    infile,
    outfile,
    n_pairs: Optional[Union[int, List[int]]] = None,
    suffix: Optional[Union[str, List[str]]] = None,
):
    """
    Rank every gene pair of an embedding by cosine similarity.
    """
    if os.path.exists(outfile):
        print(f"load existing {outfile}")
        sim_df = pd.read_csv(outfile, sep="\t", header=None)
    else:
        df = pd.read_csv(infile, sep="\t", index_col=0)
        sim_df = pd.DataFrame(
            cosine_similarity(df),
            index=df.index,
            columns=df.index,
        )
        del df
        sim_df.index.name = None
        sim_df.columns.name = None
        sim_df = sim_df.stack().reset_index()
        sim_df.columns = ["GeneA", "GeneB", "Score"]
        sim_df = sim_df[sim_df["GeneA"] < sim_df["GeneB"]]
        sim_df = sim_df.sort_values("Score", ascending=False)
        sim_df["ppi"] = sim_df["GeneA"] + "__" + sim_df["GeneB"]
        sim_df = sim_df[["ppi", "Score"]]
        sim_df.to_csv(outfile, sep="\t", index=False, header=False)

    n_pairs_list = _as_list(n_pairs)
    suffix_list = _as_list(suffix)
    if not suffix_list:
        suffix_list = [None] * len(n_pairs_list)
    if len(suffix_list) != len(n_pairs_list):
        raise ValueError(
            f"{len(n_pairs_list)} n_pairs against "
            f"{len(suffix_list)} suffix"
        )

    for n_pair, sfx in zip(n_pairs_list, suffix_list):
        if sfx is None:
            sfx = f"top{n_pair}"
        sim_df.iloc[:int(n_pair)].to_csv(
            f"{outfile}.{sfx}",
            sep="\t",
            index=False,
            header=False,
        )


def concat_emb(
    dfs: List[pd.DataFrame],
    distance: Literal["l1", "l2"] = "l1",
    fill: Literal["medoid", "medoid_jitter", "zero"] = "medoid",
    jitter_frac: float = 0.05,
    seed: Optional[int] = None,
    return_mask: bool = False,
):
    """
    Concatenate per-modality embeddings over the union of genes.
    """
    if fill not in ("medoid", "medoid_jitter", "zero"):
        raise ValueError(
            "fill must be 'medoid', 'medoid_jitter' or 'zero'"
        )
    if distance not in ("l1", "l2"):
        raise ValueError("distance must be 'l1' or 'l2'")
    rng = np.random.default_rng(seed)

    all_genes = sorted(set().union(*(df.index for df in dfs)))

    aligned = []
    observed = []
    for df in dfs:
        # the medoid is the real row closest to the per-column median
        X = df.values
        m = df.median(axis=0).values.reshape(1, -1)
        if distance == "l1":
            dists = np.abs(X - m).sum(axis=1)
        else:
            dists = np.sqrt(((X - m) ** 2).sum(axis=1))
        medoid_row = df.iloc[int(np.argmin(dists))]

        df_re = df.reindex(all_genes)
        missing_mask = df_re.isna().all(axis=1)
        if missing_mask.any():
            n_missing = int(missing_mask.sum())
            if fill == "zero":
                filler = np.zeros((n_missing, df_re.shape[1]))
            else:
                filler = np.tile(
                    medoid_row.values,
                    (n_missing, 1),
                ).astype(float)
                if fill == "medoid_jitter":
                    spread = df.std(axis=0).values.reshape(1, -1)
                    filler = filler + rng.normal(
                        loc=0.0,
                        scale=jitter_frac * spread,
                        size=filler.shape,
                    )
            df_re.loc[missing_mask, :] = filler
        aligned.append(df_re)
        observed.append(~missing_mask)

    out = pd.concat(aligned, axis=1)
    out.columns = range(out.shape[1])
    out.index.name = None
    if not return_mask:
        return out

    mask = pd.concat(observed, axis=1)
    mask.columns = range(mask.shape[1])
    mask.index.name = None
    return out, mask


def _pca_scores(X, n_pcs):
    """
    Leading principal-component scores of a column-centered matrix.
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_pcs] * s[:n_pcs]


def _cca(X, Y, eps=1e-10):
    """
    Exact CCA of two column-centered matrices by SVD.
    """
    Ux, sx, Vxt = np.linalg.svd(X, full_matrices=False)
    Uy, sy, Vyt = np.linalg.svd(Y, full_matrices=False)
    kx = sx > sx[0] * eps
    ky = sy > sy[0] * eps
    P, rho, Qt = np.linalg.svd(Ux[:, kx].T @ Uy[:, ky])
    r = min(kx.sum(), ky.sum())
    Wx = Vxt[kx].T @ (P[:, :r] / sx[kx][:, None])
    Wy = Vyt[ky].T @ (Qt[:r].T / sy[ky][:, None])
    return np.clip(rho[:r], 0, 1), Wx, Wy


def gene_specificity_cca(
    A,
    D,
    corr_threshold=0.3,
    energy_alpha=0.9,
    n_pcs=None,
):
    """
    Split two aligned gene x dim matrices into shared and unique parts.
    """
    Az = StandardScaler().fit_transform(np.asarray(A, float))
    Dz = StandardScaler().fit_transform(np.asarray(D, float))
    if n_pcs is not None:
        Az = _pca_scores(Az, n_pcs)
        Dz = _pca_scores(Dz, n_pcs)

    rho, Wa, Wd = _cca(Az, Dz)

    # keep the components above the correlation threshold that also
    # fall within the leading energy_alpha share of squared correlation
    keep = rho >= corr_threshold
    if energy_alpha is not None:
        energy = rho ** 2
        order = np.argsort(-energy)
        cum_energy = np.cumsum(energy[order]) / (energy.sum() + 1e-12)
        k_energy = np.searchsorted(cum_energy, energy_alpha) + 1
        keep_energy = np.zeros_like(keep)
        keep_energy[order[:k_energy]] = True
        keep &= keep_energy
    k = max(1, keep.sum())

    # project each side onto its own orthonormalized shared basis
    Qa, _ = np.linalg.qr(Wa[:, keep])
    Qd, _ = np.linalg.qr(Wd[:, keep])
    A_shared = Az @ Qa @ Qa.T
    D_shared = Dz @ Qd @ Qd.T
    A_unique = Az - A_shared
    D_unique = Dz - D_shared

    A_energy = (Az ** 2).sum(axis=1) + 1e-12
    D_energy = (Dz ** 2).sum(axis=1) + 1e-12
    A_unique_fraction = 1 - (A_shared ** 2).sum(axis=1) / A_energy
    D_unique_fraction = 1 - (D_shared ** 2).sum(axis=1) / D_energy
    A_unique_norm = np.linalg.norm(A_unique, axis=1)
    D_unique_norm = np.linalg.norm(D_unique, axis=1)

    shared_diff = np.linalg.norm(D_shared - A_shared, axis=1)
    A_diff = np.sqrt(shared_diff ** 2 + A_unique_norm ** 2)
    D_diff = np.sqrt(shared_diff ** 2 + D_unique_norm ** 2)
    total_diff = np.sqrt(
        shared_diff ** 2 + A_unique_norm ** 2 + D_unique_norm ** 2
    )

    return dict(
        rho=rho,
        k=k,
        D_unique_fraction=D_unique_fraction,
        D_unique_norm=D_unique_norm,
        A_unique_fraction=A_unique_fraction,
        A_unique_norm=A_unique_norm,
        total_diff=total_diff,
        A_diff=A_diff,
        D_diff=D_diff,
        A_unique=A_unique,
        D_unique=D_unique,
        A_shared=A_shared,
        D_shared=D_shared,
    )


def _read_emb(emb: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    A path or an already-loaded frame to a gene x dim frame.
    """
    if isinstance(emb, pd.DataFrame):
        return emb
    return pd.read_csv(emb, sep="\t", index_col=0)


SCORE_KEYS = {
    "unique_norm_sum": dict(
        base=lambda out: out["A_unique_norm"] + out["D_unique_norm"],
        query=lambda out: out["A_unique_norm"] + out["D_unique_norm"],
    ),
    "total_diff": dict(
        base=lambda out: out["total_diff"],
        query=lambda out: out["total_diff"],
    ),
    "unique_norm": dict(
        base=lambda out: out["A_unique_norm"],
        query=lambda out: out["D_unique_norm"],
    ),
    "unique_fraction": dict(
        base=lambda out: out["A_unique_fraction"],
        query=lambda out: out["D_unique_fraction"],
    ),
    "diff": dict(
        base=lambda out: out["A_diff"],
        query=lambda out: out["D_diff"],
    ),
}


def _rank_score(genes, values):
    """
    Per-gene scores to a gene/score/rank frame, rank 1 = most specific.
    """
    df = pd.DataFrame({"gene": genes, "score": values})
    df = df.sort_values("score", ascending=False)
    df = df.reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def cca_specific_score(
    emb_base: Union[str, pd.DataFrame],
    emb_query: Union[str, pd.DataFrame],
    outfile_base: str,
    outfile_query: str,
    score: str = "unique_norm_sum",
    corr_threshold: float = 0.3,
    energy_alpha: Optional[float] = 0.9,
    n_pcs: Optional[int] = None,
    overwrite: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Rank proteins by context-specific effect from one CCA between two
    context embeddings.
    """
    if score not in SCORE_KEYS:
        raise ValueError(f"score must be one of {sorted(SCORE_KEYS)}")

    cached = all(
        os.path.exists(path) for path in (outfile_base, outfile_query)
    )
    if cached and not overwrite:
        print(f"load existing {outfile_base} and {outfile_query}")
        return dict(
            base=pd.read_csv(outfile_base, sep="\t"),
            query=pd.read_csv(outfile_query, sep="\t"),
        )

    emb_base = _read_emb(emb_base)
    emb_query = _read_emb(emb_query)
    if emb_base.shape[1] != emb_query.shape[1]:
        raise ValueError(
            f"base is {emb_base.shape[1]}-dim against "
            f"query {emb_query.shape[1]}-dim"
        )
    genes = sorted(set(emb_base.index) & set(emb_query.index))
    if len(genes) < 3:
        raise ValueError(
            f"{len(genes)} genes shared between the two embeddings"
        )
    print(
        f"{len(genes)} genes shared, of {len(emb_base)} base "
        f"and {len(emb_query)} query"
    )

    out = gene_specificity_cca(
        emb_base.loc[genes],
        emb_query.loc[genes],
        corr_threshold=corr_threshold,
        energy_alpha=energy_alpha,
        n_pcs=n_pcs,
    )
    n_kept = out["k"]
    n_comp = len(out["rho"])
    print(f"{n_kept} of {n_comp} canonical components kept as shared")

    dfs = {}
    outfiles = {"base": outfile_base, "query": outfile_query}
    for side, outfile in outfiles.items():
        dfs[side] = _rank_score(genes, SCORE_KEYS[score][side](out))
        outdir = os.path.dirname(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        dfs[side].to_csv(outfile, sep="\t", index=False)
    return dfs


def uniprot_mapping(
    ids: List[str],
    from_db: str = "UniProtKB_AC-ID",
    to_db: str = "Gene_Name",
) -> Dict[str, str]:
    """
    Map identifiers through the UniProt ID mapping REST API.
    """
    ids = sorted(set(ids))
    job_id = submit_id_mapping(from_db=from_db, to_db=to_db, ids=ids)
    if not check_id_mapping_results_ready(job_id):
        raise RuntimeError(
            f"UniProt ID mapping job {job_id} returned no results"
        )
    link = get_id_mapping_results_link(job_id)
    results = get_id_mapping_results_search(link)

    mapping = {}
    n_extra = 0
    for record in results["results"]:
        if record["from"] in mapping:
            n_extra += 1
            continue
        mapping[record["from"]] = record["to"]

    n_failed = len(results.get("failedIds") or [])
    print(
        f"UniProt {from_db} -> {to_db}: {len(mapping)}/{len(ids)} ids "
        f"mapped, {n_failed} failed, {n_extra} extra targets dropped "
        "from one-to-many ids"
    )
    return mapping
