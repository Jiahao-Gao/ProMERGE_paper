from .base_training import (
    load_base,
    make_base,
    train_base,
)
from .data import (
    AnchorSet,
    ContextData,
    anchor_batch,
    dense_batch,
    mean_paired_cosine,
    procrustes_align,
)
from .diagnostics import (
    CollapseGuard,
    LatentGeometry,
    geometry_record,
    nan_mean,
    report_epoch,
    scalar,
    summarise,
    worst_top3,
)
from .inference import (
    encode_base,
    encode_query,
    reconstruct_base,
    reconstruct_query,
)
from .losses import (
    cosine_distance,
    gate_penalty,
    mean_offset_penalty,
    mean_terms,
    triplet_loss,
    variance_penalty,
    variance_target,
)
from .model import BaseCoEmbed, QueryCoEmbed
from .outputs import (
    average_frames,
    specific_score_frame,
    support_frame,
    write_context,
    write_frame,
    write_latents,
    write_reconstructions,
    write_snapshot,
)
from .query_training import train_query
from .training_utils import (
    epoch_batches,
    ramp_weight,
    set_seed,
    snapshot_due,
)
