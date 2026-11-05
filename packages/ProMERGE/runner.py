import os

import pandas as pd
import torch

from promerge import (
    AnchorSet,
    ContextData,
    encode_base,
    encode_query,
    reconstruct_base,
    reconstruct_query,
    set_seed,
    train_base,
    train_query,
    write_context,
    write_snapshot,
)
from promerge.data import align_to_base


DISENTANGLE_METHODS = ("CLUB", "MINE", "subtract")


def parse_embedding_paths(paths, contexts, modalities):
    """
    Map {MODALITY}-{CONTEXT}.tsv paths to {(modality, context): path}.

    The basename is split on its last hyphen and both halves must match
    the given lists exactly.
    """
    resolved = {}
    for path in paths:
        stem = os.path.basename(path)
        if stem.endswith(".tsv"):
            stem = stem[:-len(".tsv")]
        if "-" not in stem:
            raise ValueError(
                f"{path}: basename must be {{MODALITY}}-{{CONTEXT}}.tsv"
            )
        modality, context = stem.rsplit("-", 1)
        if modality not in modalities:
            raise ValueError(
                f"{path}: modality {modality!r} is not one of "
                f"{modalities}"
            )
        if context not in contexts:
            raise ValueError(
                f"{path}: context {context!r} is not one of "
                f"{contexts}"
            )
        if (modality, context) in resolved:
            raise ValueError(
                f"{path}: duplicate input for {modality}-{context}"
            )
        resolved[(modality, context)] = path
    return resolved


class ProteinCoEmbeddingGenerator:
    def __init__(
        self,
        embeddings=None,
        contexts=None,
        modalities=None,
        outdir=None,
        dimensions=128,

        n_epochs=900,
        batch_size=64,
        learn_rate=1e-3,
        hidden_size_1=512,
        hidden_size_2=256,
        dropout=0.0,
        l2_norm=True,
        triplet_margin=0.5,
        grad_clip=5.0,
        disentangle_method="CLUB",
        save_update_epochs=False,
        save_epoch=300,
        seed=None,

        lambda_reconstruction=1.0,
        lambda_replay=5.0,
        lambda_triplet=0.2,
        lambda_shared_anchor=1.0,
        lambda_mi=0.1,
        lambda_var=1.0,
        lambda_mean=1.0,
        lambda_gate=0.03,
    ):
        if not embeddings:
            raise ValueError("embeddings is required")
        if not contexts or len(contexts) < 2:
            raise ValueError(
                "contexts needs the base first and at least one query"
            )
        if not modalities:
            raise ValueError("modalities is required")
        if disentangle_method not in DISENTANGLE_METHODS:
            raise ValueError(
                "disentangle_method must be one of "
                f"{DISENTANGLE_METHODS}"
            )
        if save_epoch < 1:
            raise ValueError(
                "save_epoch is an epoch interval, so it must be "
                "at least 1"
            )

        self._embeddings = list(embeddings)
        self._contexts = list(contexts)
        self._modalities = list(modalities)
        self._outdir = outdir
        self._seed = seed
        self._params = {
            "latent_dim": dimensions,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learn_rate": learn_rate,
            "hidden_size_1": hidden_size_1,
            "hidden_size_2": hidden_size_2,
            "dropout": dropout,
            "l2_norm": l2_norm,
            "triplet_margin": triplet_margin,
            "grad_clip": grad_clip,
            "lambda_reconstruction": lambda_reconstruction,
            "lambda_replay": lambda_replay,
            "lambda_triplet": lambda_triplet,
            "lambda_shared_anchor": lambda_shared_anchor,
            "lambda_mi": lambda_mi,
            "lambda_var": lambda_var,
            "lambda_mean": lambda_mean,
            "lambda_gate": lambda_gate,
            "disentangle_method": disentangle_method,
            "save_update_epochs": save_update_epochs,
            "save_epoch": save_epoch,
        }

    def get_next_embedding(self):
        """
        Train the base context, then each query context against it,
        and yield [protein, v1, ..., vd] rows of every query's combined
        latent in order.
        """
        set_seed(self._seed)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        base = self._contexts[0]
        queries = self._contexts[1:]
        method = self._params["disentangle_method"]

        resolved = parse_embedding_paths(
            self._embeddings,
            self._contexts,
            self._modalities,
        )
        for context in self._contexts:
            available = [
                mod for mod in self._modalities
                if (mod, context) in resolved
                and os.path.isfile(resolved[(mod, context)])
            ]
            for mod in self._modalities:
                if mod in available:
                    continue
                if not available:
                    raise ValueError(
                        f"{context}-{mod} is missing. To continue "
                        f"running, provide input data for {context}; "
                        "no same-context modalities are available "
                        "for imputation."
                    )
                sources = ", ".join(
                    f"{context}-{source}" for source in available
                )
                raise ValueError(
                    f"{context}-{mod} is missing. To continue running, "
                    "at least a simple imputation by adding noise to "
                    f"{sources} should be given."
                )
        frames = {
            key: pd.read_csv(path, sep="\t", index_col=0)
            for key, path in resolved.items()
        }

        print(
            f"ProMERGE on {device}, base {base}, queries {queries}, "
            f"disentangle by {method}"
        )

        if method == "MINE":
            lambda_mi = self._params["lambda_mi"]
        for context in self._contexts:
            sizes = []
            for mod in self._modalities:
                if (mod, context) in frames:
                    rows, cols = frames[(mod, context)].shape
                    sizes.append(f"{mod} {rows}x{cols}")
            label = ", ".join(sizes) if sizes else "no inputs"
            print(f"  {context}: {label}")

        base_frames = {
            mod: frames[(mod, base)]
            for mod in self._modalities
        }

        # stage 1, the anchor
        base_data = ContextData(
            base,
            base_frames,
            device=device,
        )
        n_base = len(base_data.proteins)

        print(f"stage 1: base context {base}")

        def snapshot_base(model, epoch):
            write_snapshot(
                outdir=self._outdir,
                data=base_data,
                model=model,
                epoch=epoch,
                latents=encode_base(model, base_data),
                specifics=None,
                l2_norm=self._params["l2_norm"],
            )

        base_model, base_log = train_base(
            base_data,
            self._params,
            device,
            snapshot=snapshot_base,
        )
        base_latents = encode_base(base_model, base_data)
        reconstructions = reconstruct_base(base_model, base_data)
        write_context(
            outdir=self._outdir,
            data=base_data,
            model=base_model,
            log=base_log,
            latents=base_latents,
            reconstructions=reconstructions,
            specifics=None,
            base_proteins=None,
            l2_norm=self._params["l2_norm"],
        )

        base_proteins = set(base_data.proteins)
        base_tensors = {
            mod: torch.tensor(
                frame.to_numpy(dtype="float32"),
                device=device,
            )
            for mod, frame in base_latents.items()
        }

        for context in queries:
            query_frames = {
                mod: frames[(mod, context)]
                for mod in self._modalities
            }


            query_frames = align_to_base(
                context,
                base_frames,
                query_frames,
                latent_dim=self._params["latent_dim"],
            )
            query_data = ContextData(
                context,
                query_frames,
                device=device,
            )
            anchors = AnchorSet(
                frames={
                    mod: base_latents[mod]
                    for mod in query_data.modalities
                    if mod in base_latents
                },
                proteins=query_data.proteins,
                latent_dim=self._params["latent_dim"],
                device=device,
            )
            print(
                f"stage 2: query context {context}"
            )

            reusable = []
            for mod in query_data.modalities:
                width = query_data.input_dims[mod]
                if base_data.input_dims.get(mod) == width:
                    reusable.append(mod)

            def snapshot_query(
                model,
                epoch,
                query_data=query_data,
                anchors=anchors,
            ):
                latents, specifics = encode_query(
                    model=model,
                    data=query_data,
                    anchors=anchors,
                    latent_dim=self._params["latent_dim"],
                    device=device,
                )
                write_snapshot(
                    outdir=self._outdir,
                    data=query_data,
                    model=model,
                    epoch=epoch,
                    latents=latents,
                    specifics=specifics,
                    l2_norm=self._params["l2_norm"],
                )

            query_model, query_log = train_query(
                data=query_data,
                anchors=anchors,
                base_decoders={
                    mod: base_model.decoders[mod]
                    for mod in reusable
                },
                replay={
                    mod: (base_data.x[mod], base_tensors[mod])
                    for mod in reusable
                },
                params=self._params,
                device=device,
                snapshot=snapshot_query,
            )
            query_latents, query_specifics = encode_query(
                model=query_model,
                data=query_data,
                anchors=anchors,
                latent_dim=self._params["latent_dim"],
                device=device,
            )
            reconstructions = reconstruct_query(
                model=query_model,
                data=query_data,
                anchors=anchors,
                latent_dim=self._params["latent_dim"],
                device=device,
            )
            combined = write_context(
                outdir=self._outdir,
                data=query_data,
                model=query_model,
                log=query_log,
                latents=query_latents,
                reconstructions=reconstructions,
                specifics=query_specifics,
                base_proteins=base_proteins,
                l2_norm=self._params["l2_norm"],
            )

            values = combined.to_numpy()
            for position, protein in enumerate(combined.index):
                row = [protein]
                row.extend(values[position])
                yield row
