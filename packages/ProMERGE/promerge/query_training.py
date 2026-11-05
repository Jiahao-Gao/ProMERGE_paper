import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import anchor_batch
from .diagnostics import (
    CollapseGuard,
    LatentGeometry,
    geometry_record,
    nan_mean,
    report_epoch,
    scalar,
    summarise,
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
from .query_model import QueryCoEmbed
from .training_utils import (
    epoch_batches,
    ramp_weight,
    snapshot_due,
)


def train_query(
    data,
    anchors,
    base_decoders,
    replay,
    params,
    device,
    snapshot=None,
    gate_ramp_floor=0.1,
    gate_target=0.9,
    gate_control_rate=1.0,
    gate_lambda_min=1e-4,
    gate_lambda_max=1.0,
    club_learn_rate=None,
    club_warmup_epochs=20,
    club_grad_clip=5.0,
    club_min_batch=4,
    mine_learn_rate=None,
    mine_warmup_epochs=20,
    mine_grad_clip=5.0,
    mine_min_batch=8,
    mine_steps_per_batch=3,
    ramp_epochs=60,
):
    """
    Stage 2: encode a query context against the frozen base anchors.
    """
    subtract = params["disentangle_method"] == "subtract"
    mine = params["disentangle_method"] == "MINE"
    if gate_target is not None and not 0.0 < gate_target < 1.0:
        raise ValueError(
            "gate_target is a mean sigmoid, so it must be in (0, 1)"
        )
    if mine_steps_per_batch < 1:
        raise ValueError("mine_steps_per_batch must be at least 1")
    if mine_min_batch < 4:
        raise ValueError(
            "mine_min_batch below 4 leaves log E[e^T] estimated "
            "from too few samples for the bound to mean anything"
        )
    if club_learn_rate is None:
        club_learn_rate = 0.5 * params["learn_rate"]
    if mine_learn_rate is None:
        mine_learn_rate = 0.5 * params["learn_rate"]
    tune = {
        "learn_rate": mine_learn_rate if mine else club_learn_rate,
        "warmup_epochs": (
            mine_warmup_epochs if mine else club_warmup_epochs
        ),
        "grad_clip": mine_grad_clip if mine else club_grad_clip,
        "min_batch": mine_min_batch if mine else club_min_batch,
    }
    model = QueryCoEmbed(
        input_dims=data.input_dims,
        latent_dim=params["latent_dim"],
        hidden_1=params["hidden_size_1"],
        hidden_2=params["hidden_size_2"],
        dropout=params["dropout"],
        l2_norm=params["l2_norm"],
        base_decoders=base_decoders,
        disentangle_method=params["disentangle_method"],
    ).to(device)

    mi_weight = {}
    candidates = [] if subtract else data.modalities
    for mod in candidates:
        if mod not in anchors.modalities:
            continue
        mi_weight[mod] = 1.0
        if mine:
            model.add_mine(modality=mod)
        else:
            model.add_club(modality=mod)
    model = model.to(device)

    main_params = model.main_parameters()
    optimizer = torch.optim.Adam(main_params, lr=params["learn_rate"])
    estimators = model.estimators()
    mi_optimizer = None
    if len(mi_weight) > 0:
        mi_optimizer = torch.optim.Adam(
            estimators.parameters(),
            lr=tune["learn_rate"],
        )

    latent_dim = params["latent_dim"]
    var_target = variance_target(latent_dim, params["l2_norm"])
    guard = CollapseGuard(
        data.modalities,
        watch_gate=not subtract,
    )
    controlled = gate_target is not None
    if controlled:
        gate_lambda = gate_lambda_min
    else:
        gate_lambda = params["lambda_gate"]
    records = []

    for epoch in range(params["n_epochs"]):
        model.train()
        acc = defaultdict(list)
        geometry = {
            mod: LatentGeometry(latent_dim, device)
            for mod in data.modalities
        }

        # ramp of MI let reconstruction and anchor set the geometry first
        ramp = ramp_weight(epoch, ramp_epochs)
        if controlled:
            gate_weight = gate_lambda
        else:
            floor = gate_ramp_floor
            gate_weight = params["lambda_gate"] * max(floor, ramp)
        
        batches = epoch_batches(
            len(data.proteins),
            params["batch_size"],
            device,
        )
        for batch in batches:
            anchor = {}
            anchor_mask = {}
            for mod in data.modalities:
                anchor[mod], anchor_mask[mod] = anchor_batch(
                    anchors,
                    mod,
                    batch,
                    latent_dim,
                    device,
                )

            x = {}
            mask = {}
            predicted = {}
            direction = {}
            gate = {}
            latent = {}
            gate_logit = {}
            for mod in data.modalities:
                x[mod], mask[mod] = data.batch_inputs(mod, batch)
                encoded = model.encode(
                    mod,
                    x[mod],
                    anchor[mod],
                    anchor_mask[mod],
                )
                predicted[mod], direction[mod], gate[mod] = encoded[:3]
                latent[mod], gate_logit[mod] = encoded[4], encoded[5]

            # fit the MI estimator's own network
            if mi_optimizer is not None:
                steps = mine_steps_per_batch if mine else 1
                for _ in range(steps):
                    aux = []
                    for mod in mi_weight:
                        both = mask[mod] & anchor_mask[mod]
                        if int(both.sum()) < tune["min_batch"]:
                            continue
                        z = direction[mod][both].detach()
                        s = anchor[mod][both]
                        if mine:
                            value = model.mines[mod].estimate(
                                z,
                                s,
                                update_ema=True,
                            )
                            aux.append(-value)
                        else:
                            club = model.clubs[mod]
                            loglik = club.loglik(z, s).mean()
                            aux.append(-loglik / latent_dim)
                    if not aux:
                        break
                    mi_optimizer.zero_grad(set_to_none=True)
                    torch.stack(aux).mean().backward()
                    nn.utils.clip_grad_norm_(
                        estimators.parameters(),
                        tune["grad_clip"],
                    )
                    mi_optimizer.step()

            mi = []
            for mod in mi_weight:
                both = mask[mod] & anchor_mask[mod]
                if int(both.sum()) < tune["min_batch"]:
                    continue
                z = direction[mod][both]
                s = anchor[mod][both]
                if mine:
                    value = model.mines[mod].estimate(
                        z,
                        s,
                        update_ema=False,
                    )
                    term = F.relu(value) # floored at zero
                    acc[f"{mod}_mine"].append(scalar(term))
                else:
                    bound, _, _ = model.clubs[mod].bound(z, s)
                    acc[f"{mod}_club"].append(scalar(bound))
                    term = bound / latent_dim
                if epoch >= tune["warmup_epochs"]:
                    mi.append(mi_weight[mod] * term)

            # the shared head predicts the anchor
            shared_anchor = []
            for mod in data.modalities:
                both = mask[mod] & anchor_mask[mod]
                if not bool(both.any()):
                    continue
                gap = cosine_distance(
                    predicted[mod][both],
                    anchor[mod][both],
                )
                shared_anchor.append(gap.mean())

            # reconstruction
            recon = []
            for source in data.modalities:
                for target in data.modalities:
                    both = mask[source] & mask[target]
                    if not bool(both.any()):
                        continue
                    out = model.decode(target, latent[source])
                    term = cosine_distance(out[both], x[target][both])
                    term = term.mean()
                    recon.append(term)
                    acc[f"{target}_recon"].append(scalar(term))

            # replay
            # the decoder turn a base anchor back into the base input
            replay_terms = []
            for mod, (base_x, base_s) in replay.items():
                n_rows = min(params["batch_size"], len(base_x))
                rows = torch.randint(
                    len(base_x),
                    (n_rows,),
                    device=device,
                )
                replayed = model.replay_latent(mod, base_s[rows])
                out = model.replay(mod, replayed)
                term = cosine_distance(out, base_x[rows]).mean()
                replay_terms.append(term)
                acc[f"{mod}_replay"].append(scalar(term))

            # triplet align modalities
            triplet = []
            for first in data.modalities:
                for second in data.modalities:
                    if first == second:
                        continue
                    both = mask[first] & mask[second]
                    if not bool(both.any()):
                        continue
                    negatives, neg_positions = data.sample_negatives(
                        second,
                        batch[both],
                    )
                    neg_anchor, neg_has = anchor_batch(
                        anchors,
                        second,
                        neg_positions,
                        latent_dim,
                        device,
                    )
                    neg_direction = model.encode(
                        second,
                        negatives,
                        neg_anchor,
                        neg_has,
                    )[1]
                    pair_weight = gate[first][both] * gate[second][both]
                    pair_weight = pair_weight.detach()
                    triplet.append(
                        triplet_loss(
                            direction[first][both],
                            direction[second][both],
                            neg_direction,
                            params["triplet_margin"],
                            pair_weight.squeeze(-1),
                        )
                    )

            # gate penalty
            variance = []
            offset = []
            gate_size = []
            gate_open = []
            gate_value = []
            for mod in data.modalities:
                if not bool(mask[mod].any()):
                    continue
                chosen = mask[mod]
                if not subtract:
                    logits = gate_logit[mod][chosen]
                    gate_size.append(gate_penalty(logits))
                    opened = torch.sigmoid(logits).mean()
                    gate_open.append(float(opened.detach()))
                gate_value.append(scalar(gate[mod][chosen].mean()))
                if int(chosen.sum()) >= 2:
                    picked = latent[mod][chosen]
                    term = variance_penalty(picked, var_target)
                    variance.append(term)
                    offset.append(mean_offset_penalty(picked))
                    geometry[mod].update(picked)

            recon_term = mean_terms(recon, device)
            replay_term = mean_terms(replay_terms, device)
            triplet_term = mean_terms(triplet, device)
            shared_term = mean_terms(shared_anchor, device)
            mi_term = mean_terms(mi, device)
            variance_term = mean_terms(variance, device)
            offset_term = mean_terms(offset, device)
            gate_term = mean_terms(gate_size, device)
            total = (
                params["lambda_reconstruction"] * recon_term
                + params["lambda_replay"] * replay_term
                + params["lambda_shared_anchor"] * shared_term
                + params["lambda_var"] * variance_term
                + params["lambda_mean"] * offset_term
                + (0.0 if subtract else gate_weight) * gate_term
                + ramp * params["lambda_triplet"] * triplet_term
                + ramp * params["lambda_mi"] * mi_term
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(main_params, params["grad_clip"])
            optimizer.step()
            if mi_optimizer is not None:
                mi_optimizer.zero_grad(set_to_none=True)

            acc["total"].append(scalar(total))
            acc["recon"].append(scalar(recon_term))
            acc["replay"].append(scalar(replay_term))
            acc["triplet"].append(scalar(triplet_term))
            acc["shared_anchor"].append(scalar(shared_term))
            acc["mi"].append(scalar(mi_term))
            acc["var"].append(scalar(variance_term))
            acc["mean_offset"].append(scalar(offset_term))
            acc["gate"].append(nan_mean(gate_value))
            if not subtract:
                acc["gate_prior"].append(scalar(gate_term))
                acc["gate_weight"].append(gate_weight)
                acc["gate_controlled"].append(float(controlled))
                acc["gate_open"].append(nan_mean(gate_open))
            acc["ramp"].append(ramp)
            for mod in data.modalities:
                if bool(mask[mod].any()):
                    chosen = mask[mod]
                    gate_mean = gate[mod][chosen].mean()
                    acc[f"{mod}_gate"].append(scalar(gate_mean))

        record = summarise(epoch, acc)
        record.update(geometry_record(geometry))
        if not subtract:
            record["gate_lambda"] = gate_lambda
        records.append(record)
        report_epoch(data.context, epoch, record, params)
        if snapshot is not None and snapshot_due(params, epoch):
            snapshot(model, epoch)
        guard.check(record, epoch, data.context)

        # update gate-penalty weight
        if controlled and not subtract:
            error = record.get("gate_open", float("nan"))
            error = error - gate_target
            if math.isfinite(error):
                scaled = gate_lambda * math.exp(
                    gate_control_rate * error
                )
                gate_lambda = float(
                    np.clip(
                        scaled,
                        gate_lambda_min,
                        gate_lambda_max,
                    )
                )

    return model, pd.DataFrame(records)
