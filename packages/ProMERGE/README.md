# ProMERGE

ProMERGE learns protein embeddings across data modalities and biological
contexts. It trains a base context first, then uses its per-modality
embeddings as frozen anchors for query contexts. It produces a single 
protein embedding, and query-specific effect vectors and scores.


## Requirements and setup
We have tested the package with the following setup:  
- `python==3.11.15`
- `torch==2.11.0+cu126`
- `numpy==1.26.4`
- `pandas==2.3.3`

To run the code:
```python
import sys
sys.path.insert(0, "./ProMERGE")
from runner import ProteinCoEmbeddingGenerator
```

Device is selected automatically: CUDA when available, otherwise CPU


## Input format

Supply one tab-separated file for every requested modality-context pair.
Each file must have a header row, protein identifiers in the first
column, and numeric embedding dimensions in the remaining columns:

```text
protein\tdim_1\tdim_2\tdim_3
ACKR5\t0.213176\t0.183766\t0.131074
CCR1\t0.108717\t0.174962\t0.338818
```

File basenames should be `{MODALITY}-{CONTEXT}.tsv`, such as
`BioPlex-HCT116.tsv`. Context names cannot contain hyphens.

## Quick start

```python
from runner import ProteinCoEmbeddingGenerator

model = ProteinCoEmbeddingGenerator(
    embeddings=[
        "input/BioPlex-HCT116.tsv",
        "input/SECMX-HCT116.tsv",
        "input/BioPlex-HEK293T.tsv",
        "input/SECMX-HEK293T.tsv",
    ],
    contexts=["HCT116", "HEK293T"],
    modalities=["BioPlex", "SECMX"],
    outdir="output/promerge",
    dimensions=128,
    disentangle_method="CLUB",
    seed=1,
)

for row in model.get_next_embedding():
    pass
```

`contexts[0]` is the base; subsequent contexts are trained independently against that base. The iterator must be consumed for training and writing to occur. Each yielded row is `[protein, v1, ..., vd]` from a query's combined embedding.


## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `embeddings` | `None` | Required list of input TSV paths |
| `contexts` | `None` | Required ordered context names: base first, then at least one query |
| `modalities` | `None` | Required modality names |
| `outdir` | `None` | Required output directory |
| `dimensions` | `128` | Latent embedding width |
| `save_update_epochs` | `False` | Write intermediate latent and model snapshots |
| `save_epoch` | `300` | Snapshot interval, at least 1; includes epoch 0 |
| `seed` | `None` | Random seed for python, numpy and torch |
| `n_epochs` | `900` | Number of training epochs |
| `batch_size` | `64` | Number of proteins per training batch |
| `learn_rate` | `1e-3` | Main Adam optimizer learning rate |
| `hidden_size_1` | `512` | First encoder hidden width. |
| `hidden_size_2` | `256` | Second encoder hidden width. |
| `dropout` | `0.0` | Dropout probability in encoder/decoder MLPs |
| `l2_norm` | `True` | Normalize base and final query latents embeddings |
| `triplet_margin` | `0.5` | Cosine triplet-loss margin. |
| `disentangle_method` | `"CLUB"` | One of `"CLUB"`, `"MINE"`, or `"subtract"`, case-sensitive. |

To adjust the network structure and more parameters, please refere to the source code.

## Output files

Files are written under `{outdir}/{context}/`.

| File | Contents |
| --- | --- |
| `{ctx}_latent.tsv` | Per-protein mean latent across present modalities |
| `{ctx}_{modality}_latent.tsv` | Per-protein latent for that modality |
| `{ctx}_specific.tsv` | Query only: mean specific vector across present modalities |
| `{ctx}_specific_score.tsv` | Query only: per-modality specific-vector norms, `n_modality`, and their mean `specific_score` |
| `{ctx}_support.tsv` | Per-modality presence flags |
| `{ctx}_{source}___{target}_reconstructed.tsv` | Reconstruction of target using source modality input |
| `{ctx}_model.pth` | PyTorch model `state_dict`|
| `{ctx}_log.tsv` | Per-epoch losses and other statistics |