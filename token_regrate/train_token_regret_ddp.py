import argparse
import json
import math
import os
import sys

from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import open_clip

# Make project root importable when launched as token_regrate/train_token_regret_ddp.py via torchrun.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_VQ, get_masking_ratio, open_clip_text_encoding

from token_regrate.config import get_config
from token_regrate.dataset import *
from token_regrate.utils import *


@torch.no_grad()
def forward_maskgen(
    model,
    input_ids,
    condition,
    condition_pooled,
    sample_aesthetic_score=6.5,
    ratio=None,
    guidance_scale=12.0,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    none_condition=None,
    none_condition_pooled=None,
):
    """Forward MaskGen and optionally apply the same CFG logits used during sampling."""

    def _forward_once(ids, text_condition, pooled_condition):
        aesthetic_score = None
        if sample_aesthetic_score is not None:
            aesthetic_score = torch.full((ids.shape[0],), sample_aesthetic_score, device=ids.device)

        embeddings = model.embeddings(ids)
        cond = model.text_embed_proj(text_condition)

        if model.micro_condition:
            pooled_condition = model.concat_micro_cond(pooled_condition, aesthetic_score)
        pooled_condition = model.cond_pooled_proj(pooled_condition)

        x = embeddings + model.pos_embed[:, :embeddings.shape[1]]
        for blk in model.blocks:
            cond, x = blk(x, cond, pooled_condition.squeeze(1))
        x = model.norm(x, pooled_condition.squeeze(1))
        logits = model.lm_head(x)
        return logits, x, pooled_condition.squeeze(1)

    use_cfg = none_condition is not None and none_condition_pooled is not None
    if use_cfg:
        cfg_ratio = 1.0 if ratio is None else ratio
        if isinstance(cfg_ratio, torch.Tensor):
            cfg_ratio = float(cfg_ratio.flatten()[0].item())
        cfg_scale = _compute_cfg_scale(
            ratio=cfg_ratio,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            device=input_ids.device,
        )
        if cfg_scale != 0.0:
            num_samples = input_ids.shape[0]
            logits_all, hidden_all, text_all = _forward_once(
                torch.cat([input_ids, input_ids], dim=0),
                torch.cat([condition, none_condition], dim=0),
                torch.cat([condition_pooled, none_condition_pooled], dim=0),
            )
            cond_logits, uncond_logits = logits_all[:num_samples], logits_all[num_samples:]
            logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            return logits, hidden_all[:num_samples], text_all[:num_samples]

    return _forward_once(input_ids, condition, condition_pooled)


def select_all_token_positions(candidate_mask):
    """Return every token position, with validity controlled by candidate_mask."""
    bsz, seq_len = candidate_mask.shape
    idx = torch.arange(seq_len, device=candidate_mask.device).unsqueeze(0).expand(bsz, -1)
    return idx, candidate_mask.bool()


def prepare_text_guidance(text, clip_tokenizer, clip_encoder, device):
    """Encode prompts with CLIP text tower for tokenizer decode guidance."""
    text_tokens = clip_tokenizer(text).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_embed = clip_encoder.token_embedding(text_tokens).to(cast_dtype)
    text_embed = text_embed + clip_encoder.positional_embedding.to(cast_dtype)
    text_embed = text_embed.permute(1, 0, 2)
    text_embed = clip_encoder.transformer(text_embed, attn_mask=clip_encoder.attn_mask)
    text_embed = text_embed.permute(1, 0, 2)
    text_guidance = clip_encoder.ln_final(text_embed)
    return text_guidance.to(device)


def _resolve_num_selected(seq_len, ratio_or_count, min_tokens=1):
    """Resolve a ratio-or-count setting into an integer token count."""
    value = float(ratio_or_count)
    if value <= 0:
        return 0
    if value <= 1:
        count = int(seq_len * value)
    else:
        count = int(value)
    count = max(int(min_tokens), count)
    return min(int(seq_len), count)


def select_topk_token_positions(scores, remask_ratio, candidate_mask=None, min_tokens=1):
    """Select top-k token indices by critic score with optional candidate mask."""
    bsz, seq_len = scores.shape
    k = _resolve_num_selected(seq_len, remask_ratio, min_tokens=min_tokens)
    if k == 0:
        return torch.zeros(bsz, 0, dtype=torch.long, device=scores.device), torch.zeros(bsz, 0, dtype=torch.bool, device=scores.device)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    elif candidate_mask.dtype != torch.bool:
        candidate_mask = candidate_mask.bool()
    masked_scores = scores.masked_fill(~candidate_mask, float("-inf"))
    idx = masked_scores.topk(k=k, dim=-1).indices
    valid = candidate_mask.gather(dim=-1, index=idx)
    return idx, valid


def remask_positions(tokens, indices, mask_token_id, valid_mask=None):
    """Replace selected token positions with mask token where valid."""
    out = tokens.clone()
    if valid_mask is None:
        valid_mask = torch.ones_like(indices, dtype=torch.bool)
    gathered = out.gather(dim=-1, index=indices)
    masked = torch.full_like(gathered, int(mask_token_id))
    update = torch.where(valid_mask, masked, gathered)
    return out.scatter(dim=-1, index=indices, src=update)


def _add_gumbel_noise(logits, temperature):
    """Add Gumbel noise to logits for stochastic argmax sampling."""
    def log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -log(-log(noise))

    g = gumbel_noise(logits)
    return logits + temperature * g


@torch.no_grad()
def compute_token_nll(logits, gt_tokens):
    """Compute token-level NLL against ground-truth image tokens."""
    return F.cross_entropy(logits.transpose(1, 2), gt_tokens, reduction="none")


def _canonical_counterfactual_utility(name):
    """Normalize counterfactual target names used by configs and notebooks."""
    name = str(name or "token_ce").strip().lower()
    aliases = {
        "full": "full_sequence_ce",
        "sequence": "full_sequence_ce",
        "sequence_ce": "full_sequence_ce",
        "full_ce": "full_sequence_ce",
        "full_sequence_ce": "full_sequence_ce",
        "token": "token_ce",
        "token_ce": "token_ce",
        "local": "local_window_ce",
        "local_ce": "local_window_ce",
        "window": "local_window_ce",
        "window_ce": "local_window_ce",
        "patch": "local_window_ce",
        "patch_ce": "local_window_ce",
        "neighborhood": "local_window_ce",
        "neighborhood_ce": "local_window_ce",
        "local_window_ce": "local_window_ce",
    }
    if name not in aliases:
        raise ValueError(
            f"Unsupported counterfactual_utility={name!r}. "
            "Expected token_ce, local_window_ce, or full_sequence_ce."
        )
    return aliases[name]


def _reduce_counterfactual_utility(token_nll, selected_idx, counterfactual_utility, window_radius=0):
    """Reduce token NLL to the utility attached to each selected intervention."""
    utility = _canonical_counterfactual_utility(counterfactual_utility)
    if utility == "full_sequence_ce":
        return token_nll.mean(dim=-1, keepdim=True).expand_as(selected_idx)
    if utility == "token_ce":
        return token_nll.gather(dim=-1, index=selected_idx)

    radius = max(0, int(window_radius))
    if radius == 0:
        return token_nll.gather(dim=-1, index=selected_idx)

    bsz, seq_len = token_nll.shape
    select_count = selected_idx.shape[1]
    offsets = torch.arange(-radius, radius + 1, device=token_nll.device)
    window_idx = selected_idx.unsqueeze(-1) + offsets.view(1, 1, -1)
    in_bounds = window_idx.ge(0) & window_idx.lt(seq_len)
    safe_idx = window_idx.clamp(min=0, max=seq_len - 1)
    gathered = token_nll.gather(dim=-1, index=safe_idx.reshape(bsz, -1))
    gathered = gathered.reshape(bsz, select_count, -1)
    weights = in_bounds.to(dtype=token_nll.dtype)
    return (gathered * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)


def transform_regret_targets(regrets, transform="tanh", valid_mask=None, eps=1e-6):
    """Apply a stable regression target transform while preserving token ordering."""
    name = str(transform or "none").strip().lower()
    if name in {"none", "identity", "raw"}:
        return regrets
    if name in {"tanh", "squash"}:
        return torch.tanh(regrets)
    if name in {"zscore", "standardize"}:
        if valid_mask is None:
            keep = torch.ones_like(regrets, dtype=torch.bool)
        else:
            keep = valid_mask.bool()
        if keep.sum() < 2:
            return regrets - regrets[keep].mean() if keep.any() else regrets
        vals = regrets[keep]
        mean = vals.mean()
        std = vals.std(unbiased=False).clamp(min=float(eps))
        return (regrets - mean) / std
    raise ValueError(
        f"Unsupported regret_target_transform={name!r}. "
        "Expected none, tanh, or zscore."
    )


@torch.no_grad()
def compute_counterfactual_regret(
    model,
    gt_tokens,
    z_t,
    condition,
    condition_pooled,
    timesteps,
    selected_idx,
    selected_valid,
    step_indices=None,
    sample_aesthetic_score=6.5,
    counterfactual_chunk_size=64,
    counterfactual_rollout_steps=1,
    counterfactual_utility="token_ce",
    counterfactual_window_radius=0,
    num_sample_steps=16,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    softmax_temperature_annealing=True,
    refine_softmax_temperature=0.7,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=False,
    none_condition=None,
    none_condition_pooled=None,
):
    """Compute CE regret after remasking token i and rolling through plain MaskGen steps.

    The current rollout path uses one shared start step for the whole batch and never applies critic remasking.
    """
    debug_cf = os.environ.get("TOKEN_REGRET_DEBUG_COUNTERFACTUAL", "0").lower() in {"1", "true", "yes", "on"}
    debug_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    debug_max_chunks = int(os.environ.get("TOKEN_REGRET_DEBUG_COUNTERFACTUAL_CHUNKS", "3"))
    debug_preview = int(os.environ.get("TOKEN_REGRET_DEBUG_COUNTERFACTUAL_PREVIEW", "8"))

    def _debug_enabled():
        return debug_cf and debug_rank == 0

    def _debug(message):
        if _debug_enabled():
            print(f"[counterfactual-debug rank={debug_rank}] {message}")

    def _debug_tensor(name, value):
        if not _debug_enabled():
            return
        if value is None:
            _debug(f"{name}: None")
            return
        t = value.detach()
        if t.numel() == 0:
            _debug(f"{name}: empty shape={tuple(t.shape)} dtype={t.dtype}")
            return
        if torch.is_floating_point(t):
            finite = torch.isfinite(t)
            finite_count = int(finite.sum().item())
            nan_count = int(torch.isnan(t).sum().item())
            inf_count = int(torch.isinf(t).sum().item())
            if finite_count > 0:
                finite_t = t[finite].float()
                _debug(
                    f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"finite={finite_count}/{t.numel()} nan={nan_count} inf={inf_count} "
                    f"min={finite_t.min().item():.6g} max={finite_t.max().item():.6g} "
                    f"mean={finite_t.mean().item():.6g} std={finite_t.std(unbiased=False).item():.6g}"
                )
            else:
                _debug(
                    f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"finite=0/{t.numel()} nan={nan_count} inf={inf_count}"
                )
            return

        unique_vals, unique_counts = torch.unique(t, return_counts=True)
        preview = list(zip(unique_vals[:debug_preview].tolist(), unique_counts[:debug_preview].tolist()))
        _debug(
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"unique_count={unique_vals.numel()} preview={preview}"
        )

    def _debug_token_delta(name, before, after):
        if not _debug_enabled():
            return
        before_t = before.detach()
        after_t = after.detach()
        changed = before_t.ne(after_t)
        changed_count = int(changed.sum().item())
        total = int(changed.numel())
        mask_token_id = int(model.mask_token_id)
        _debug(
            f"{name}: changed={changed_count}/{total} "
            f"before_masks={int(before_t.eq(mask_token_id).sum().item())} "
            f"after_masks={int(after_t.eq(mask_token_id).sum().item())}"
        )
        if changed_count > 0:
            changed_pos = torch.nonzero(changed, as_tuple=False)[:debug_preview].tolist()
            _debug(f"{name}: changed_pos_preview={changed_pos}")

    def _debug_regret_signs(name, values):
        if not _debug_enabled():
            return
        vals = values.detach()
        if vals.numel() == 0:
            _debug(f"{name}: no values")
            return
        finite = torch.isfinite(vals)
        vals = vals[finite]
        if vals.numel() == 0:
            _debug(f"{name}: no finite values")
            return
        _debug(
            f"{name}: positive={int(vals.gt(0).sum().item())} "
            f"negative={int(vals.lt(0).sum().item())} "
            f"zero={int(vals.eq(0).sum().item())}"
        )

    if step_indices is None:
        raise ValueError("compute_counterfactual_regret requires step_indices from build_rollout_state.")

    utility_name = _canonical_counterfactual_utility(counterfactual_utility)
    window_radius = max(0, int(counterfactual_window_radius))
    start_step = int(step_indices[0].item())
    _debug(
        f"start_step={start_step} num_sample_steps={int(num_sample_steps)} "
        f"rollout_steps={int(counterfactual_rollout_steps)} chunk_size={int(counterfactual_chunk_size)} "
        f"utility={utility_name} window_radius={window_radius} repair_greedy={bool(repair_greedy)}"
    )
    _debug_tensor("gt_tokens", gt_tokens)
    _debug_tensor("z_t", z_t)
    _debug_tensor("timesteps", timesteps)
    _debug_tensor("step_indices", step_indices)
    _debug_tensor("selected_idx", selected_idx)
    _debug_tensor("selected_valid", selected_valid)

    baseline_tokens, target_ratio = _rollout_maskgen_steps(
        model=model,
        token_ids=z_t,
        condition=condition,
        condition_pooled=condition_pooled,
        start_step=start_step,
        num_sample_steps=num_sample_steps,
        rollout_steps=counterfactual_rollout_steps,
        guidance_scale=guidance_scale,
        randomize_temperature=randomize_temperature,
        sample_aesthetic_score=sample_aesthetic_score,
        softmax_temperature_annealing=softmax_temperature_annealing,
        refine_softmax_temperature=refine_softmax_temperature,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=guidance_decay_scale_pow,
        prob_sorting=prob_sorting,
        repair_greedy=repair_greedy,
        none_condition=none_condition,
        none_condition_pooled=none_condition_pooled,
    )
    _debug(f"baseline target_ratio={float(target_ratio):.6g}")
    _debug_token_delta("z_t -> baseline_tokens", z_t, baseline_tokens)
    _debug_tensor("baseline_tokens", baseline_tokens)

    baseline_logits, _, _ = forward_maskgen(
        model=model,
        input_ids=baseline_tokens,
        condition=condition,
        condition_pooled=condition_pooled,
        ratio=target_ratio,
        guidance_scale=guidance_scale,
        sample_aesthetic_score=sample_aesthetic_score,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=guidance_decay_scale_pow,
        none_condition=none_condition,
        none_condition_pooled=none_condition_pooled,
    )
    _debug_tensor("baseline_logits", baseline_logits)

    baseline_token_nll = compute_token_nll(baseline_logits, gt_tokens)
    baseline_selected = _reduce_counterfactual_utility(
        baseline_token_nll,
        selected_idx,
        utility_name,
        window_radius,
    ).to(dtype=baseline_token_nll.dtype)
    _debug_tensor("baseline_token_nll", baseline_token_nll)
    _debug_tensor("baseline_selected_utility", baseline_selected)

    counterfactual_regrets = torch.zeros_like(baseline_selected)
    pair = torch.nonzero(selected_valid, as_tuple=False)
    _debug(f"valid_selected_pairs={int(pair.shape[0])}")
    if pair.numel() == 0:
        _debug("No valid selected positions; returning all-zero counterfactual regrets.")
        return counterfactual_regrets

    chunk_size = max(1, int(counterfactual_chunk_size))
    mask_token_id = int(model.mask_token_id)
    for start in range(0, pair.shape[0], chunk_size):
        chunk_id = start // chunk_size
        debug_chunk = _debug_enabled() and chunk_id < max(0, debug_max_chunks)
        part = pair[start:start + chunk_size]
        if part.numel() == 0:
            continue
        b_idx = part[:, 0]
        k_idx = part[:, 1]
        tok_idx = selected_idx[b_idx, k_idx]
        if debug_chunk:
            _debug(
                f"chunk={chunk_id} pair_range=[{start}, {start + part.shape[0]}) "
                f"chunk_size={int(part.shape[0])}"
            )
            _debug(f"chunk={chunk_id} b_idx_preview={b_idx[:debug_preview].tolist()}")
            _debug(f"chunk={chunk_id} k_idx_preview={k_idx[:debug_preview].tolist()}")
            _debug(f"chunk={chunk_id} tok_idx_preview={tok_idx[:debug_preview].tolist()}")

        z_cf = z_t[b_idx].clone()
        z_cf[torch.arange(z_cf.shape[0], device=z_cf.device), tok_idx] = mask_token_id
        if debug_chunk:
            _debug_token_delta(f"chunk={chunk_id} z_t[b_idx] -> z_cf", z_t[b_idx], z_cf)
            _debug_tensor(f"chunk={chunk_id} z_cf", z_cf)

        cf_tokens, cf_ratio = _rollout_maskgen_steps(
            model=model,
            token_ids=z_cf,
            condition=condition[b_idx],
            condition_pooled=condition_pooled[b_idx],
            start_step=start_step,
            num_sample_steps=num_sample_steps,
            rollout_steps=counterfactual_rollout_steps,
            guidance_scale=guidance_scale,
            randomize_temperature=randomize_temperature,
            sample_aesthetic_score=sample_aesthetic_score,
            softmax_temperature_annealing=softmax_temperature_annealing,
            refine_softmax_temperature=refine_softmax_temperature,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            prob_sorting=prob_sorting,
            repair_greedy=repair_greedy,
            none_condition=none_condition[b_idx] if none_condition is not None else None,
            none_condition_pooled=none_condition_pooled[b_idx] if none_condition_pooled is not None else None,
        )
        if debug_chunk:
            _debug(f"chunk={chunk_id} cf_ratio={float(cf_ratio):.6g}")
            _debug_token_delta(f"chunk={chunk_id} z_cf -> cf_tokens", z_cf, cf_tokens)
            _debug_token_delta(f"chunk={chunk_id} baseline_tokens[b_idx] -> cf_tokens", baseline_tokens[b_idx], cf_tokens)
            _debug_tensor(f"chunk={chunk_id} cf_tokens", cf_tokens)

        logits_cf, _, _ = forward_maskgen(
            model=model,
            input_ids=cf_tokens,
            condition=condition[b_idx],
            condition_pooled=condition_pooled[b_idx],
            ratio=cf_ratio,
            guidance_scale=guidance_scale,
            sample_aesthetic_score=sample_aesthetic_score,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            none_condition=none_condition[b_idx] if none_condition is not None else None,
            none_condition_pooled=none_condition_pooled[b_idx] if none_condition_pooled is not None else None,
        )
        if debug_chunk:
            _debug_tensor(f"chunk={chunk_id} logits_cf", logits_cf)

        cf_token_nll = compute_token_nll(logits_cf, gt_tokens[b_idx])
        cf_selected = _reduce_counterfactual_utility(
            cf_token_nll,
            tok_idx.unsqueeze(-1),
            utility_name,
            window_radius,
        ).squeeze(-1)
        chunk_regret = baseline_selected[b_idx, k_idx] - cf_selected
        if debug_chunk:
            _debug_tensor(f"chunk={chunk_id} baseline_selected_utility", baseline_selected[b_idx, k_idx])
            _debug_tensor(f"chunk={chunk_id} cf_token_nll", cf_token_nll)
            _debug_tensor(f"chunk={chunk_id} cf_selected_utility", cf_selected)
            _debug_tensor(f"chunk={chunk_id} regret", chunk_regret)
            _debug_regret_signs(f"chunk={chunk_id} regret_signs", chunk_regret)
        counterfactual_regrets[b_idx, k_idx] = chunk_regret

    valid_regrets = counterfactual_regrets[selected_valid]
    _debug_tensor("counterfactual_regrets", counterfactual_regrets)
    _debug_tensor("counterfactual_regrets[selected_valid]", valid_regrets)
    _debug_regret_signs("counterfactual_regrets[selected_valid] signs", valid_regrets)

    return counterfactual_regrets


@torch.no_grad()
def masked_pearson_corr(x, y, valid_mask, eps=1e-8):
    """Compute a masked Pearson correlation for critic-vs-regret sanity checks."""
    keep = valid_mask.bool()
    if keep.sum() < 2:
        return x.new_tensor(0.0)
    x_keep = x[keep].float()
    y_keep = y[keep].float()
    x_keep = x_keep - x_keep.mean()
    y_keep = y_keep - y_keep.mean()
    denom = torch.sqrt(x_keep.pow(2).mean() * y_keep.pow(2).mean()).clamp(min=float(eps))
    return (x_keep * y_keep).mean() / denom


def gap_weighted_pairwise_rank_loss(scores, regrets, valid_mask, margin=0.05, gap_threshold=0.0, eps=1e-8):
    """Rank higher-regret tokens above lower-regret tokens, weighted by regret gap."""
    gap_threshold = float(max(0.0, gap_threshold))
    losses = []
    for b in range(scores.shape[0]):
        keep = valid_mask[b].bool()
        if int(keep.sum().item()) < 2:
            continue
        score = scores[b][keep]
        regret = regrets[b][keep].detach()
        regret_gap = regret[:, None] - regret[None, :]
        pair_weight = regret_gap.clamp(min=0.0)
        if gap_threshold > 0.0:
            pair_weight = pair_weight * regret_gap.gt(gap_threshold).to(dtype=pair_weight.dtype)
        if not pair_weight.gt(0.0).any():
            continue
        score_gap = score[:, None] - score[None, :]
        weighted_loss = F.relu(float(margin) - score_gap) * pair_weight
        losses.append(weighted_loss.sum() / pair_weight.sum().clamp(min=float(eps)))

    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def _resolve_refine_loop_count(refine_loops, refine_start_step, num_sample_steps):
    """Clamp refine loops to timesteps where a critic remask decision is made."""
    max_decision_loops = max(0, num_sample_steps - refine_start_step - 1)
    if max_decision_loops == 0:
        return 0
    if refine_loops is None:
        return max_decision_loops
    requested_loops = int(refine_loops)
    if requested_loops <= 0:
        return max_decision_loops
    return min(requested_loops, max_decision_loops)


def _resolve_schedule_remask_count(model, ratio, budget_fraction=1.0):
    """Use MaskGen's schedule to decide the critic remask budget."""
    if isinstance(ratio, torch.Tensor):
        ratio = float(ratio.flatten()[0].item())
    ratio = float(max(1e-6, min(1.0, float(ratio))))
    budget_fraction = float(max(0.0, budget_fraction))
    schedule_mask_ratio = float(get_masking_ratio(ratio, model.mask_schedule_strategy))
    budget = int(math.floor(int(model.image_seq_len) * schedule_mask_ratio * budget_fraction))
    return max(0, min(int(model.image_seq_len), budget))


def _resolve_future_rollout_count(start_step, num_sample_steps, rollout_steps):
    """Resolve a target rollout horizon. Use -1 to roll to the final sample step."""
    start_step = int(start_step)
    num_sample_steps = int(num_sample_steps)
    remaining_steps = max(0, num_sample_steps - start_step)
    if remaining_steps == 0:
        return 0
    requested_steps = 1 if rollout_steps is None else int(rollout_steps)
    if requested_steps < 0:
        return remaining_steps
    return min(remaining_steps, max(1, requested_steps))


def _compute_cfg_scale(ratio, guidance_scale, guidance_decay, guidance_decay_scale_pow, device):
    """Compute per-step CFG scale following the selected decay schedule."""
    ratio = float(max(1e-6, min(1.0, float(ratio))))
    if guidance_decay == "none":
        return float(guidance_scale)
    if guidance_decay == "cosine":
        scale_pow = torch.ones((1), device=device) * float(guidance_decay_scale_pow)
        scale_step = (1.0 - torch.cos((ratio ** scale_pow) * torch.pi)) * 0.5
        return float(((float(guidance_scale) - 1.0) * scale_step + 1.0).item())
    if guidance_decay == "flippedcosine":
        scale_pow = torch.ones((1), device=device) * float(guidance_decay_scale_pow)
        scale_step = torch.cos((ratio ** scale_pow) * torch.pi) * 0.5
        return float(((float(guidance_scale) - 1.0) * scale_step + 1.0).item())
    if guidance_decay == "linear":
        return ratio * (float(guidance_scale) - 1.0) + 1.0
    raise ValueError(f"Unsupported guidance_decay={guidance_decay}")


@torch.no_grad()
def maskgen_denoise_step(
    model,
    token_ids,
    condition,
    condition_pooled,
    ratio,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    sample_aesthetic_score=6.5,
    softmax_temperature_annealing=True,
    refine_softmax_temperature=0.7,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=False,
    none_condition=None,
    none_condition_pooled=None,
    final_step=False,
):
    """Run one MaskGen denoising step and return the next partially masked state."""
    debug_denoise = os.environ.get("TOKEN_REGRET_DEBUG_DENOISE", "0").lower() in {"1", "true", "yes", "on"}
    debug_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _debug_tensor(name, value):
        if not debug_denoise or debug_rank != 0:
            return
        t = value.detach()
        if t.numel() == 0:
            print(f"[denoise-debug rank={debug_rank}] {name}: empty shape={tuple(t.shape)} dtype={t.dtype}")
            return
        if torch.is_floating_point(t):
            finite = torch.isfinite(t)
            finite_count = int(finite.sum().item())
            nan_count = int(torch.isnan(t).sum().item())
            inf_count = int(torch.isinf(t).sum().item())
            if finite_count > 0:
                finite_t = t[finite].float()
                print(
                    f"[denoise-debug rank={debug_rank}] {name}: "
                    f"shape={tuple(t.shape)} dtype={t.dtype} finite={finite_count}/{t.numel()} "
                    f"nan={nan_count} inf={inf_count} "
                    f"min={finite_t.min().item():.6g} max={finite_t.max().item():.6g} "
                    f"mean={finite_t.mean().item():.6g} std={finite_t.std(unbiased=False).item():.6g}"
                )
            else:
                print(
                    f"[denoise-debug rank={debug_rank}] {name}: "
                    f"shape={tuple(t.shape)} dtype={t.dtype} finite=0/{t.numel()} "
                    f"nan={nan_count} inf={inf_count}"
                )
            return

        unique_vals, unique_counts = torch.unique(t, return_counts=True)
        preview = list(zip(unique_vals[:8].tolist(), unique_counts[:8].tolist()))
        print(
            f"[denoise-debug rank={debug_rank}] {name}: "
            f"shape={tuple(t.shape)} dtype={t.dtype} unique_count={unique_vals.numel()} "
            f"preview={preview}"
        )

    annealed_temp = float(randomize_temperature) * (1.0 - ratio)
    mask_token_id = int(model.mask_token_id)
    is_mask = token_ids.eq(mask_token_id)
    logits, _, _ = forward_maskgen(
        model=model,
        input_ids=token_ids,
        condition=condition,
        condition_pooled=condition_pooled,
        ratio=ratio,
        guidance_scale=guidance_scale,
        sample_aesthetic_score=sample_aesthetic_score,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=guidance_decay_scale_pow,
        none_condition=none_condition,
        none_condition_pooled=none_condition_pooled,
    )
    _debug_tensor("condition", condition)
    _debug_tensor("condition_pooled", condition_pooled)
    if none_condition is not None:
        _debug_tensor("none_condition", none_condition)
    if none_condition_pooled is not None:
        _debug_tensor("none_condition_pooled", none_condition_pooled)
    _debug_tensor("input_token_ids", token_ids)
    _debug_tensor("is_mask", is_mask)
    _debug_tensor("raw_logits", logits)

    if bool(softmax_temperature_annealing):
        softmax_temperature = 0.5 + 0.8 * (1.0 - ratio)
    else:
        softmax_temperature = max(float(refine_softmax_temperature), 1e-6)
    logits_for_sample = logits / float(softmax_temperature)
    if 0 <= mask_token_id < logits_for_sample.shape[-1]:
        logits_for_sample[..., mask_token_id] = -1e9
    _debug_tensor("logits_for_sample", logits_for_sample)

    noisy_logits = None
    if bool(repair_greedy):
        proposed_ids = logits_for_sample.argmax(dim=-1)
    else:
        noisy_logits = _add_gumbel_noise(logits_for_sample, annealed_temp)
        _debug_tensor("noisy_logits", noisy_logits)
        proposed_ids = noisy_logits.argmax(dim=-1)

    proposed_logits = logits_for_sample.gather(dim=-1, index=proposed_ids.unsqueeze(-1)).squeeze(-1)
    sampled_ids = torch.where(is_mask, proposed_ids, token_ids)
    _debug_tensor("proposed_ids", proposed_ids)
    _debug_tensor("proposed_logits", proposed_logits)
    _debug_tensor("sampled_ids", sampled_ids)
    if bool(final_step):
        return sampled_ids

    sampled_logits = torch.where(is_mask, proposed_logits, torch.full_like(proposed_logits, float("inf")))
    schedule_mask_ratio = float(get_masking_ratio(ratio, model.mask_schedule_strategy))
    num_samples = token_ids.shape[0]
    schedule_mask_len = torch.full(
        (num_samples,),
        int(math.floor(int(model.image_seq_len) * schedule_mask_ratio)),
        dtype=torch.long,
        device=token_ids.device,
    )
    remaining_masks = is_mask.sum(dim=-1).long()
    mask_len = torch.where(
        remaining_masks > 0,
        torch.maximum(
            torch.ones_like(remaining_masks),
            torch.minimum(remaining_masks - 1, schedule_mask_len),
        ),
        torch.zeros_like(remaining_masks),
    )

    confidence = sampled_logits
    if bool(prob_sorting) and not bool(repair_greedy):
        confidence = _add_gumbel_noise(confidence, annealed_temp)
    _debug_tensor("sampled_logits", sampled_logits)
    _debug_tensor("confidence", confidence)
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    safe_k = mask_len.clamp(min=1)
    cut_off = sorted_confidence.gather(dim=-1, index=(safe_k - 1).unsqueeze(-1))
    masking = (confidence <= cut_off) & mask_len.unsqueeze(-1).gt(0)
    _debug_tensor("sorted_confidence", sorted_confidence)
    _debug_tensor("cut_off", cut_off)
    _debug_tensor("masking", masking)
    if debug_denoise and debug_rank == 0:
        print(
            f"[denoise-debug rank={debug_rank}] "
            f"ratio={float(ratio):.6g} annealed_temp={annealed_temp:.6g} "
            f"softmax_temperature={float(softmax_temperature):.6g} "
            f"repair_greedy={bool(repair_greedy)} prob_sorting={bool(prob_sorting)} "
            f"final_step={bool(final_step)} mask_token_id={mask_token_id}"
        )
        print(
            f"[denoise-debug rank={debug_rank}] "
            f"remaining_masks={remaining_masks[:4].tolist()} "
            f"schedule_mask_len={schedule_mask_len[:4].tolist()} "
            f"mask_len={mask_len[:4].tolist()} "
            f"masking_count={int(masking.sum().item())}"
        )

    return torch.where(masking, mask_token_id, sampled_ids)


@torch.no_grad()
def _rollout_maskgen_steps(
    model,
    token_ids,
    condition,
    condition_pooled,
    start_step,
    num_sample_steps,
    rollout_steps,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    sample_aesthetic_score=6.5,
    softmax_temperature_annealing=True,
    refine_softmax_temperature=0.7,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=False,
    none_condition=None,
    none_condition_pooled=None,
):
    """Roll token ids forward through normal MaskGen transitions and return final ids plus the last ratio."""
    step_count = _resolve_future_rollout_count(start_step, num_sample_steps, rollout_steps)
    ids = token_ids
    ratio = float(max(1e-6, min(1.0, float(int(start_step) + 1) / float(num_sample_steps))))
    for step in range(int(start_step), min(int(num_sample_steps), int(start_step) + step_count)):
        ratio = float(step + 1) / float(num_sample_steps)
        ratio = float(max(1e-6, min(1.0, ratio)))
        ids = maskgen_denoise_step(
            model=model,
            token_ids=ids,
            condition=condition,
            condition_pooled=condition_pooled,
            ratio=ratio,
            guidance_scale=guidance_scale,
            randomize_temperature=randomize_temperature,
            sample_aesthetic_score=sample_aesthetic_score,
            softmax_temperature_annealing=softmax_temperature_annealing,
            refine_softmax_temperature=refine_softmax_temperature,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            prob_sorting=prob_sorting,
            repair_greedy=repair_greedy,
            none_condition=none_condition,
            none_condition_pooled=none_condition_pooled,
            final_step=step == int(num_sample_steps) - 1,
        )
    return ids, ratio


@torch.no_grad()
def critic_remask_tokens(
    model,
    critic,
    token_ids,
    condition,
    condition_pooled,
    ratio,
    guidance_scale=12.0,
    sample_aesthetic_score=6.5,
    remask_ratio=0.05,
    critic_use_hidden=True,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    none_condition=None,
    none_condition_pooled=None,
):
    """Apply critic top-k remasking. `remask_ratio` may be a ratio or an absolute schedule budget."""
    if critic is None:
        return token_ids
    
    logits, hidden, text_feat = forward_maskgen(
        model=model,
        input_ids=token_ids,
        condition=condition,
        condition_pooled=condition_pooled,
        ratio=ratio,
        guidance_scale=guidance_scale,
        sample_aesthetic_score=sample_aesthetic_score,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=guidance_decay_scale_pow,
        none_condition=none_condition,
        none_condition_pooled=none_condition_pooled,
    )
    timesteps = torch.full((token_ids.shape[0],), ratio, device=token_ids.device)
    selection_scores = critic(
        hidden_states=hidden if bool(critic_use_hidden) else None,
        logits=logits,
        timesteps=timesteps,
        text_features=text_feat,
    )
    select_idx, select_valid = select_topk_token_positions(
        selection_scores,
        remask_ratio,
        candidate_mask=token_ids.ne(int(model.mask_token_id)),
        min_tokens=1,
    )
    return remask_positions(token_ids, select_idx, int(model.mask_token_id), valid_mask=select_valid)


def unwrap_ddp(module):
    """Return the underlying module when a model is wrapped in DDP."""
    return module.module if isinstance(module, DDP) else module


def copy_critic_weights_(target, source):
    """Initialize a frozen rollout critic from the train critic."""
    target.load_state_dict(unwrap_ddp(source).state_dict())
    target.eval()
    target.requires_grad_(False)


@torch.no_grad()
def update_ema_critic_(target, source, decay):
    """Update the frozen rollout critic with EMA weights from the train critic."""
    decay = max(0.0, min(1.0, float(decay)))
    source_state = unwrap_ddp(source).state_dict()
    target_state = target.state_dict()
    for name, target_value in target_state.items():
        source_value = source_state[name].detach().to(device=target_value.device)
        if torch.is_floating_point(target_value):
            target_value.mul_(decay).add_(source_value.to(dtype=target_value.dtype), alpha=1.0 - decay)
        else:
            target_value.copy_(source_value.to(dtype=target_value.dtype))


@torch.no_grad()
def build_rollout_state(
    model,
    condition,
    condition_pooled,
    none_condition,
    none_condition_pooled,
    device,
    batch_size,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    sample_aesthetic_score=6.5,
    num_sample_steps=16,
    softmax_temperature_annealing=True,
    refine_softmax_temperature=0.7,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=False,
    refine_loops=1,
    refine_start_step=10,
    critic=None,
    remask_ratio=0.05,
    critic_use_hidden=True,
    fixed_step=None,
):
    """
     creates a realistic partially denoised MaskGen token sequence at a randomly chosen refinement timestep, 
     optionally simulating earlier frozen-critic remasking, 
     so the train critic can learn what action to take at that timestep.
    Build the simple CTRC training decision state:
    1) choose a refinement timestep
    2) simulate the inference policy from the all-mask state up to that timestep
    3) return the current partially masked token state before the critic action

    If `critic` is provided, it should be a frozen rollout policy such as an EMA
    target critic, not the train critic currently receiving gradients.
    """
    assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]
    
    decision_loop_count = _resolve_refine_loop_count(refine_loops, refine_start_step, num_sample_steps)
    refine_end_step = int(refine_start_step) + decision_loop_count
    if fixed_step is None or int(fixed_step) < 0:
        loop_idx = int(torch.randint(0, decision_loop_count, (1,), device=device).item())
        step = int(refine_start_step) + loop_idx
    else:
        step = max(int(refine_start_step), min(int(fixed_step), int(refine_end_step) - 1))
    ratio = float(step + 1) / float(num_sample_steps)
    timesteps = torch.full((batch_size,), ratio, device=device)
    step_indices = torch.full((batch_size,), step, dtype=torch.long, device=device)

    ids = torch.full((batch_size, int(model.image_seq_len)), int(model.mask_token_id), device=device)
    for prev_step in range(step):
        prev_ratio = float(prev_step + 1) / float(num_sample_steps)
        critic_guided_step = critic is not None and int(refine_start_step) <= prev_step < refine_end_step
        if critic_guided_step:
            critic_budget = _resolve_schedule_remask_count(model, prev_ratio, budget_fraction=remask_ratio)
            ids = critic_remask_tokens(
                model=model,
                critic=critic,
                token_ids=ids,
                condition=condition,
                condition_pooled=condition_pooled,
                ratio=prev_ratio,
                guidance_scale=guidance_scale,
                sample_aesthetic_score=sample_aesthetic_score,
                remask_ratio=critic_budget,
                critic_use_hidden=critic_use_hidden,
                guidance_decay=guidance_decay,
                guidance_decay_scale_pow=guidance_decay_scale_pow,
                none_condition=none_condition,
                none_condition_pooled=none_condition_pooled,
            )
        ids = maskgen_denoise_step(
            model=model,
            token_ids=ids,
            condition=condition,
            condition_pooled=condition_pooled,
            ratio=prev_ratio,
            guidance_scale=guidance_scale,
            randomize_temperature=randomize_temperature,
            sample_aesthetic_score=sample_aesthetic_score,
            softmax_temperature_annealing=softmax_temperature_annealing,
            refine_softmax_temperature=refine_softmax_temperature,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            prob_sorting=prob_sorting,
            repair_greedy=repair_greedy,
            none_condition=none_condition,
            none_condition_pooled=none_condition_pooled,
            final_step=False,
        )

    return ids, timesteps, step_indices


@torch.no_grad()
def generate_wrapper(
    model,
    captions,
    clip_tokenizer,
    clip_encoder,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    sample_aesthetic_score=6.5,
    num_sample_steps=16,
    use_critic_head=False,
    critic=None,
    remask_ratio=0.05,
    refine_start_step=10,
    critic_use_hidden=True,
    refine_softmax_temperature=0.7,
    softmax_temperature_annealing=True,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=False,
    remask_min_score=0.0,
    refine_loops=None,
):
    """Generate tokens with one stepwise engine, optionally inserting CTRC top-k remasking."""
    _ = remask_min_score
    assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]
    assert refine_start_step >= 0, "refine_start_step must be non-negative"
    assert refine_start_step < num_sample_steps, "refine_start_step must be less than num_sample_steps"
    if bool(use_critic_head) and critic is None:
        raise ValueError("use_critic_head=True requires a non-None critic.")

    sample_aesthetic_score = float(
        getattr(model, "sample_aesthetic_score", 6.5) if sample_aesthetic_score is None else sample_aesthetic_score
    )

    condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
    none_cond, none_cond_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
    num_samples = condition.shape[0]
    device = condition.device
    none_cond = none_cond.repeat(num_samples, 1, 1)
    none_cond_pooled = none_cond_pooled.repeat(num_samples, 1, 1)

    ids = torch.full((num_samples, int(model.image_seq_len)), int(model.mask_token_id), device=device)
    mask_token_id = int(model.mask_token_id)

    decision_loop_count = _resolve_refine_loop_count(refine_loops, refine_start_step, num_sample_steps)
    refine_end_step = int(refine_start_step) + max(0, decision_loop_count)

    for step in range(int(num_sample_steps)):
        ratio = float(step + 1) / float(num_sample_steps)
        ratio = float(max(1e-6, min(1.0, ratio)))
        critic_guided_step = bool(use_critic_head) and int(refine_start_step) <= step < refine_end_step
        if critic_guided_step:
            critic_budget = _resolve_schedule_remask_count(model, ratio, budget_fraction=remask_ratio)
            ids = critic_remask_tokens(
                model=model,
                critic=critic,
                token_ids=ids,
                condition=condition,
                condition_pooled=condition_pooled,
                ratio=ratio,
                guidance_scale=guidance_scale,
                sample_aesthetic_score=sample_aesthetic_score,
                remask_ratio=critic_budget,
                critic_use_hidden=critic_use_hidden,
                guidance_decay=guidance_decay,
                guidance_decay_scale_pow=guidance_decay_scale_pow,
                none_condition=none_cond,
                none_condition_pooled=none_cond_pooled,
            )
        ids = maskgen_denoise_step(
            model=model,
            token_ids=ids,
            condition=condition,
            condition_pooled=condition_pooled,
            ratio=ratio,
            guidance_scale=guidance_scale,
            randomize_temperature=randomize_temperature,
            sample_aesthetic_score=sample_aesthetic_score,
            softmax_temperature_annealing=softmax_temperature_annealing,
            refine_softmax_temperature=refine_softmax_temperature,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            prob_sorting=prob_sorting,
            repair_greedy=repair_greedy,
            none_condition=none_cond,
            none_condition_pooled=none_cond_pooled,
            final_step=step == int(num_sample_steps) - 1,
        )

    if ids.eq(mask_token_id).any():
        cfg_scale_last = _compute_cfg_scale(
            ratio=1.0,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            device=device,
        )
        if cfg_scale_last != 0.0:
            logits_all_last, _, _ = forward_maskgen(
                model,
                torch.cat([ids, ids], dim=0),
                torch.cat([condition, none_cond], dim=0),
                torch.cat([condition_pooled, none_cond_pooled], dim=0),
                sample_aesthetic_score=sample_aesthetic_score,
            )
            cond_logits_last, uncond_logits_last = logits_all_last[:num_samples], logits_all_last[num_samples:]
            logits_last = cond_logits_last + (cond_logits_last - uncond_logits_last) * cfg_scale_last
        else:
            logits_last, _, _ = forward_maskgen(
                model,
                ids,
                condition,
                condition_pooled,
                sample_aesthetic_score=sample_aesthetic_score,
            )
        if 0 <= mask_token_id < logits_last.shape[-1]:
            logits_last[..., mask_token_id] = -1e9
        fill_ids = logits_last.argmax(dim=-1)
        ids = torch.where(ids.eq(mask_token_id), fill_ids, ids)

    if ids.eq(mask_token_id).any():
        raise RuntimeError("Refinement left mask tokens in the final token grid.")
    return ids


@torch.no_grad()
def generate_image_vq_batch(
    prompts,
    model,
    tokenizer,
    clip_tokenizer,
    clip_encoder,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    aesthetic_score=6.5,
    num_sample_steps=16,
    use_regret_remask=False,
    critic=None,
    remask_ratio=0.05,
    refine_start_step=10,
    critic_use_hidden=True,
    refine_softmax_temperature=0.7,
    repair_greedy=False,
    remask_min_score=0.0,
    refine_loops=None,
):
    """Generate image batch from prompts with optional critic-based token refinement."""
    _ = remask_min_score
    model_device = next(model.parameters()).device
    clip_encoder = clip_encoder.to(model_device)
    tokenizer = tokenizer.to(model_device)
    if use_regret_remask and critic is None:
        raise ValueError("use_regret_remask=True but critic is None")
    if critic is not None:
        critic = critic.to(model_device)

    tokens = generate_wrapper(
        model=model,
        captions=prompts,
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        guidance_scale=float(guidance_scale),
        randomize_temperature=float(randomize_temperature),
        sample_aesthetic_score=float(aesthetic_score),
        num_sample_steps=int(num_sample_steps),
        use_critic_head=bool(use_regret_remask),
        critic=critic,
        remask_ratio=remask_ratio,
        refine_start_step=refine_start_step,
        refine_loops=refine_loops,
        critic_use_hidden=bool(critic_use_hidden),
        refine_softmax_temperature=refine_softmax_temperature,
        repair_greedy=bool(repair_greedy),
    ).to(model_device)
    text_guidance = prepare_text_guidance(prompts, clip_tokenizer, clip_encoder, model_device)
    image = tokenizer.decode_tokens(tokens, text_guidance)
    image = torch.clamp(image, 0.0, 1.0)
    image = (image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(arr) for arr in image]

def _collect_fixed_training_subset(dataset_pipeline, count, rank, world_size):
    """Collect a deterministic local subset from the configured training stream."""
    images = []
    captions = []
    target_count = max(0, int(count))
    if target_count <= 0:
        return images, captions

    for batch_images, batch_captions in dataset_pipeline.iter_batches(
        batch_size=target_count,
        rank=rank,
        world_size=world_size,
    ):
        remaining = target_count - len(captions)
        if remaining <= 0:
            break
        images.extend(batch_images[:remaining])
        captions.extend(batch_captions[:remaining])
        if len(captions) >= target_count:
            break

    if len(captions) < target_count:
        raise RuntimeError(
            f"Requested {target_count} training images on rank {rank}, "
            f"but the dataset yielded only {len(captions)} usable examples."
        )
    return images, captions


def _iter_fixed_subset_batches(images, captions, batch_size):
    """Yield batches from an in-memory fixed subset."""
    batch_size = max(1, int(batch_size))
    for start in range(0, len(captions), batch_size):
        end = start + batch_size
        yield images[start:end], captions[start:end]


def _save_fixed_training_subset(output_dir, images, captions, rank, world_size):
    """Save fixed training subset images and captions for later notebook inspection."""
    subset_dir = os.path.join(output_dir, "used_training_images")
    rank_dir = os.path.join(subset_dir, f"rank_{int(rank):04d}")
    os.makedirs(rank_dir, exist_ok=True)

    rank_manifest_path = os.path.join(subset_dir, f"rank_{int(rank):04d}.jsonl")
    rows = []
    for local_idx, (image, caption) in enumerate(zip(images, captions)):
        image_name = f"sample_{int(local_idx):05d}.png"
        rel_image = os.path.join(f"rank_{int(rank):04d}", image_name)
        image.save(os.path.join(subset_dir, rel_image))
        rows.append(
            {
                "rank": int(rank),
                "world_size": int(world_size),
                "local_index": int(local_idx),
                "global_index": int(rank) + int(local_idx) * max(1, int(world_size)),
                "image": rel_image,
                "caption": str(caption),
            }
        )

    with open(rank_manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return subset_dir, rank_manifest_path


def _merge_fixed_training_subset_manifests(output_dir, world_size):
    """Merge per-rank subset manifests into one stable manifest for notebook use."""
    subset_dir = os.path.join(output_dir, "used_training_images")
    rows = []
    for rank_idx in range(max(1, int(world_size))):
        rank_manifest_path = os.path.join(subset_dir, f"rank_{int(rank_idx):04d}.jsonl")
        if not os.path.isfile(rank_manifest_path):
            continue
        with open(rank_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    rows.sort(key=lambda row: (int(row.get("global_index", 0)), int(row.get("rank", 0)), int(row.get("local_index", 0))))
    merged_manifest_path = os.path.join(subset_dir, "manifest.jsonl")
    with open(merged_manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return merged_manifest_path, len(rows)


def apply_cli_overrides(config):
    """Apply notebook/script command-line overrides to the default ConfigDict."""
    parser = argparse.ArgumentParser(description="Train the token-regret critic.", add_help=True)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--per-gpu-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--counterfactual-chunk-size", type=int)
    parser.add_argument("--counterfactual-rollout-steps", type=int)
    parser.add_argument("--counterfactual-utility", type=str)
    parser.add_argument("--counterfactual-window-radius", type=int)
    parser.add_argument("--counterfactual-repair-greedy", action="store_true", default=None)
    parser.add_argument(
        "--no-counterfactual-repair-greedy",
        action="store_false",
        dest="counterfactual_repair_greedy",
        default=None,
    )
    parser.add_argument("--regret-target-transform", type=str)
    parser.add_argument("--target-critic-ema-decay", type=float)
    parser.add_argument("--fixed-rollout-step", type=int)
    parser.add_argument("--use-target-critic-replay", action="store_true", default=None)
    parser.add_argument(
        "--no-target-critic-replay",
        action="store_false",
        dest="use_target_critic_replay",
        default=None,
    )
    parser.add_argument("--train-repair-greedy", action="store_true", default=None)
    parser.add_argument(
        "--no-train-repair-greedy",
        action="store_false",
        dest="train_repair_greedy",
        default=None,
    )
    parser.add_argument("--lambda-rank", type=float)
    parser.add_argument("--rank-margin", type=float)
    parser.add_argument("--rank-gap-threshold", type=float)
    parser.add_argument("--train-remask-ratio", type=float)
    parser.add_argument("--train-refine-start-step", type=int)
    parser.add_argument("--refine-loops", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume-checkpoint", type=str)
    parser.add_argument("--train-data-source", type=str)
    parser.add_argument("--train-dataset-mode", type=str)
    parser.add_argument("--cc12m-cache-dir", type=str)
    parser.add_argument("--disable-cc12m-cache", action="store_true", default=None)
    parser.add_argument("--stream-prefetch-batches", type=int)
    parser.add_argument("--cc12m-loader-workers", type=int)
    parser.add_argument("--cc12m-loader-max-pending", type=int)
    args, unknown = parser.parse_known_args()

    if unknown and is_main_process():
        print(f"WARNING: ignoring unknown training arguments: {' '.join(unknown)}", file=sys.stderr)

    training = config.training
    dataset_cfg = config.dataset
    if os.environ.get("DIST_BACKEND"):
        training.dist_backend = str(os.environ["DIST_BACKEND"])

    assignments = [
        (training, "num_epochs", args.num_epochs),
        (training, "max_steps", args.max_steps),
        (training, "max_train_images", args.max_train_images),
        (training, "per_gpu_batch_size", args.per_gpu_batch_size),
        (training, "learning_rate", args.learning_rate),
        (training, "counterfactual_chunk_size", args.counterfactual_chunk_size),
        (training, "counterfactual_rollout_steps", args.counterfactual_rollout_steps),
        (training, "counterfactual_utility", args.counterfactual_utility),
        (training, "counterfactual_window_radius", args.counterfactual_window_radius),
        (training, "counterfactual_repair_greedy", args.counterfactual_repair_greedy),
        (training, "regret_target_transform", args.regret_target_transform),
        (training, "target_critic_ema_decay", args.target_critic_ema_decay),
        (training, "fixed_rollout_step", args.fixed_rollout_step),
        (training, "use_target_critic_replay", args.use_target_critic_replay),
        (training, "train_repair_greedy", args.train_repair_greedy),
        (training, "lambda_rank", args.lambda_rank),
        (training, "rank_margin", args.rank_margin),
        (training, "rank_gap_threshold", args.rank_gap_threshold),
        (training, "train_remask_ratio", args.train_remask_ratio),
        (config.inference, "remask_ratio", args.train_remask_ratio),
        (training, "train_refine_start_step", args.train_refine_start_step),
        (config.inference, "refine_start_step", args.train_refine_start_step),
        (training, "refine_loops", args.refine_loops),
        (config.inference, "refine_loops", args.refine_loops),
        (training, "save_every", args.save_every),
        (training, "log_every", args.log_every),
        (training, "seed", args.seed),
        (config.inference, "seed", args.seed),
        (config.runtime, "resume_checkpoint", args.resume_checkpoint),
        (dataset_cfg, "source", args.train_data_source),
        (dataset_cfg, "mode", args.train_dataset_mode),
        (dataset_cfg, "cc12m_cache_dir", args.cc12m_cache_dir),
        (dataset_cfg, "stream_prefetch_batches", args.stream_prefetch_batches),
        (dataset_cfg, "cc12m_loader_workers", args.cc12m_loader_workers),
        (dataset_cfg, "cc12m_loader_max_pending", args.cc12m_loader_max_pending),
    ]
    for section, key, value in assignments:
        if value is not None:
            section[key] = value

    if args.disable_cc12m_cache is not None:
        dataset_cfg.disable_cc12m_cache = bool(args.disable_cc12m_cache)

    if args.output_dir:
        output_dir = str(args.output_dir)
        config.experiment.output_dir = output_dir
        config.experiment.logging_dir = os.path.join(output_dir, "logs")
        config.logging.tensorboard_dir = os.path.join(output_dir, "logs", "tb")
        config.logging.metrics_path = os.path.join(output_dir, "logs", "metrics.jsonl")

    return config


def main():
    """Entry point for distributed token-regret critic training."""
    config = get_config()
    config = apply_cli_overrides(config)
    training = config.training
    dataset_cfg = config.dataset
    model_cfg = config.model
    logging_cfg = config.logging
    experiment_cfg = config.experiment

    device, rank, world_size, local_rank = setup_training_runtime(
        config,
        dist_backend=str(training.dist_backend),
    )
    hf_cache_dir = setup_cache_environment()
    set_global_seed(int(training.seed) + rank)

    tok_dir = os.path.join(hf_cache_dir, "turkeyju/tokenizer_tatitok_bl128_vq")
    gen_dir = os.path.join(hf_cache_dir, "turkeyju/generator_maskgen_vq_xl")
    tokenizer = TATiTok.from_pretrained(pretrained_model_name_or_path=tok_dir, cache_dir=tok_dir).to(device)
    tokenizer.eval().requires_grad_(False)
    model = MaskGen_VQ.from_pretrained(pretrained_model_name_or_path=gen_dir, cache_dir=gen_dir).to(device)
    model.eval().requires_grad_(False)
    clip_encoder, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai", force_quick_gelu=True)
    del clip_encoder.visual
    clip_encoder.transformer.batch_first = False
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval().requires_grad_(False)
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")

    critic = build_token_regret_critic(model, use_hidden=True).to(device)

    use_ddp = world_size > 1 and device.type == "cuda"
    if use_ddp:
        critic = DDP(critic, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(critic.parameters(), lr=float(training.learning_rate), weight_decay=1e-4)
    target_critic = build_token_regret_critic(model, use_hidden=True).to(device)
    copy_critic_weights_(target_critic, critic)
    target_ema_decay = max(0.0, min(1.0, float(getattr(training, "target_critic_ema_decay", 0.995))))

    start_step = 0
    resume_path = str(config.runtime.resume_checkpoint).strip()
    if resume_path:
        resume_path = os.path.expanduser(resume_path)
        if not os.path.isfile(resume_path):
            if is_main_process():
                print(
                    f"WARNING: resume checkpoint not found at '{resume_path}'. Starting from scratch.",
                    file=sys.stderr,
                )
        else:
            critic_for_load = critic.module if isinstance(critic, DDP) else critic
            try:
                ckpt = load_critic_checkpoint(
                    resume_path,
                    critic_for_load,
                    optimizer=optimizer,
                    map_location=device,
                    target_critic=target_critic,
                )
            except FileNotFoundError:
                if is_main_process():
                    print(
                        f"WARNING: resume checkpoint disappeared before loading: '{resume_path}'. Starting from scratch.",
                        file=sys.stderr,
                    )
            else:
                start_step = int(ckpt.get("step", 0))
                if not bool(ckpt.get("target_critic_restored", False)):
                    copy_critic_weights_(target_critic, critic)
                if is_main_process():
                    print(f"Resumed from checkpoint: {resume_path} (step={start_step})")
                    if bool(ckpt.get("critic_scalar_head_migrated", False)):
                        print("Migrated legacy critic head to scalar regret; optimizer state was not restored.")
                    if not bool(ckpt.get("target_critic_restored", False)):
                        print("No EMA target critic found in checkpoint; initialized target critic from online critic.")

    dataset_pipeline = TrainingDatasetPipeline(
        mode=dataset_cfg.mode,
        sources=dataset_cfg.source,
        cc12m_cache_dir=dataset_cfg.cc12m_cache_dir,
        cc12m_cache_images=not bool(dataset_cfg.disable_cc12m_cache),
        cc12m_loader_workers=int(dataset_cfg.cc12m_loader_workers),
        cc12m_max_pending=int(dataset_cfg.cc12m_loader_max_pending),
    )
    if is_main_process():
        print(f"Dataset pipeline: {dataset_pipeline.describe()}")
        if dataset_pipeline.use_cc12m:
            print(
                "CC12M image cache: "
                f"{'disabled' if not dataset_pipeline.cc12m_cache_enabled else dataset_pipeline.cc12m_cache_dir}"
            )
            print(f"CC12M loader workers: {int(dataset_pipeline.cc12m_loader_workers)}")

    dataset_examples = dataset_pipeline.infer_total_examples()
    steps_per_epoch = None
    if dataset_examples is not None:
        steps_per_epoch = max(1, math.ceil(int(dataset_examples) / (int(training.per_gpu_batch_size) * max(world_size, 1))))

    total_steps = None if steps_per_epoch is None else int(steps_per_epoch) * int(training.num_epochs)
    max_steps = int(getattr(training, "max_steps", 0))
    stop_step = int(start_step) + max_steps if max_steps > 0 else None
    if stop_step is not None:
        total_steps = stop_step
    max_train_images = max(0, int(getattr(training, "max_train_images", 0)))
    max_train_images_per_rank = 0
    fixed_subset_images = None
    fixed_subset_captions = None
    if max_train_images > 0:
        max_train_images_per_rank = max(1, int(math.ceil(max_train_images / max(1, world_size))))
        effective_max_train_images = max_train_images_per_rank * max(1, world_size)
        subset_steps_per_epoch = max(
            1,
            int(math.ceil(max_train_images_per_rank / max(1, int(training.per_gpu_batch_size)))),
        )
        subset_total_steps = int(start_step) + subset_steps_per_epoch * int(training.num_epochs)
        total_steps = subset_total_steps if stop_step is None else min(stop_step, subset_total_steps)
        if is_main_process():
            if effective_max_train_images == max_train_images:
                print(
                    f"Fixed training subset: {max_train_images} total "
                    f"({max_train_images_per_rank} per rank), replayed for {int(training.num_epochs)} epochs"
                )
            else:
                print(
                    "Fixed training subset: "
                    f"requested {max_train_images}, using {effective_max_train_images} total "
                    f"({max_train_images_per_rank} per rank) so DDP ranks stay synchronized, "
                    f"replayed for {int(training.num_epochs)} epochs"
                )

    os.makedirs(experiment_cfg.output_dir, exist_ok=True)
    if max_train_images > 0:
        fixed_subset_images, fixed_subset_captions = _collect_fixed_training_subset(
            dataset_pipeline=dataset_pipeline,
            count=max_train_images_per_rank,
            rank=rank,
            world_size=world_size,
        )
        subset_dir, _ = _save_fixed_training_subset(
            output_dir=experiment_cfg.output_dir,
            images=fixed_subset_images,
            captions=fixed_subset_captions,
            rank=rank,
            world_size=world_size,
        )
        if is_dist():
            dist.barrier()
        if is_main_process():
            merged_manifest_path, merged_count = _merge_fixed_training_subset_manifests(
                output_dir=experiment_cfg.output_dir,
                world_size=world_size,
            )
            print(f"Saved {merged_count} fixed training examples to {subset_dir}")
            print(f"Fixed training subset manifest: {merged_manifest_path}")
        if is_dist():
            dist.barrier()

    pbar = tqdm(total=total_steps, initial=start_step, desc="iterations", unit="iter", disable=not is_main_process())
    step = int(start_step)

    cfg = config.to_dict()
    train_repair_greedy = bool(
        getattr(training, "train_repair_greedy", getattr(training, "counterfactual_repair_greedy", config.inference.repair_greedy))
    )
    use_target_critic_replay = bool(getattr(training, "use_target_critic_replay", True))
    fixed_rollout_step = int(getattr(training, "fixed_rollout_step", -1))
    fixed_rollout_step_arg = fixed_rollout_step if fixed_rollout_step >= 0 else None
    tb_logger = None
    if is_main_process() and bool(logging_cfg.enabled):
        tb_logger = TensorboardLogger(log_dir=experiment_cfg.output_dir, run_name="logs", enabled=True)
        tb_logger.log_config(cfg)
        tb_logger.log_text("dataset/pipeline", dataset_pipeline.describe(), step=start_step)

    if int(dataset_cfg.stream_prefetch_batches) > 0:
        stream_prefetch_batches = int(dataset_cfg.stream_prefetch_batches)
    else:
        stream_prefetch_batches = max(1, min(8, max(2, int(training.per_gpu_batch_size) // 64)))

    for _ in range(int(training.num_epochs)):
        if max_train_images > 0:
            stream = _iter_fixed_subset_batches(
                fixed_subset_images,
                fixed_subset_captions,
                batch_size=int(training.per_gpu_batch_size),
            )
        else:
            batch_iter = dataset_pipeline.iter_batches(
                batch_size=int(training.per_gpu_batch_size),
                rank=rank,
                world_size=world_size,
            )
            stream = prefetch_batches(
                batch_iter,
                prefetch_batches=stream_prefetch_batches,
            )
        for images, captions in stream:
            if stop_step is not None and step >= stop_step:
                break
            batch_size = len(captions)
            if batch_size <= 0:
                continue
            gt_tokens = images_to_tokens(tokenizer, images).to(device, non_blocking=True)
            train_aes = float(training.train_aesthetic_score)

            with torch.no_grad():
                # Match inference-side text conditioning (no training-time text dropout).
                # This part is exactly the same as the beginning of `generate_wrapper()`; I got from MaskGen_VQ.generate() but without the generation loop.
                condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
                condition = condition.to(device, non_blocking=True)
                condition_pooled = condition_pooled.to(device, non_blocking=True)
                none_condition, none_condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
                none_condition = none_condition.repeat(batch_size, 1, 1).to(device, non_blocking=True)
                none_condition_pooled = none_condition_pooled.repeat(batch_size, 1, 1).to(device, non_blocking=True)

                z_t, timesteps, rollout_step_indices = build_rollout_state(
                    model=model,
                    condition=condition,
                    condition_pooled=condition_pooled,
                    none_condition=none_condition,
                    none_condition_pooled=none_condition_pooled,
                    device=device,
                    batch_size=batch_size,
                    guidance_scale=float(training.train_guidance_scale),
                    randomize_temperature=float(training.train_randomize_temperature),
                    sample_aesthetic_score=float(training.train_aesthetic_score),
                    num_sample_steps=int(model_cfg.sample_steps),
                    refine_softmax_temperature=0.7,
                    refine_loops=int(training.refine_loops),
                    refine_start_step=int(training.train_refine_start_step),
                    repair_greedy=train_repair_greedy,
                    # For normal policy training, replay uses the frozen EMA policy. For overfit sanity
                    # checks, disable this so the same images produce stable state distributions.
                    critic=target_critic if use_target_critic_replay else None,
                    remask_ratio=float(training.train_remask_ratio),
                    critic_use_hidden=True,
                    fixed_step=fixed_rollout_step_arg,
                )

                # Compute the original logits, which will be used as a baseline for the counterfactual regret
                logits_orig, hidden_orig, text_feat = forward_maskgen(
                    model=model,
                    input_ids=z_t,
                    condition=condition,
                    condition_pooled=condition_pooled,
                    ratio=timesteps,
                    guidance_scale=float(training.train_guidance_scale),
                    sample_aesthetic_score=train_aes,
                    guidance_decay="cosine",
                    guidance_decay_scale_pow=1.0,
                    none_condition=none_condition,
                    none_condition_pooled=none_condition_pooled,
                )

                base_candidate_mask = z_t.ne(int(model.mask_token_id))
                selected_idx, selected_valid = select_all_token_positions(base_candidate_mask)
                
                counterfactual_regrets = compute_counterfactual_regret(
                    model=model,
                    gt_tokens=gt_tokens,
                    z_t=z_t,
                    condition=condition,
                    condition_pooled=condition_pooled,
                    timesteps=timesteps,
                    step_indices=rollout_step_indices,
                    selected_idx=selected_idx,
                    selected_valid=selected_valid,
                    sample_aesthetic_score=train_aes,
                    counterfactual_chunk_size=int(training.counterfactual_chunk_size),
                    counterfactual_rollout_steps=int(getattr(training, "counterfactual_rollout_steps", 1)),
                    counterfactual_utility=str(getattr(training, "counterfactual_utility", "token_ce")),
                    counterfactual_window_radius=int(getattr(training, "counterfactual_window_radius", 0)),
                    num_sample_steps=int(model_cfg.sample_steps),
                    guidance_scale=float(training.train_guidance_scale),
                    randomize_temperature=float(training.train_randomize_temperature),
                    refine_softmax_temperature=0.7,
                    guidance_decay="cosine",
                    guidance_decay_scale_pow=1.0,
                    repair_greedy=bool(getattr(training, "counterfactual_repair_greedy", config.inference.repair_greedy)),
                    none_condition=none_condition,
                    none_condition_pooled=none_condition_pooled,
                )

            pred_regret = critic(
                hidden_states=hidden_orig,
                logits=logits_orig,
                timesteps=timesteps,
                text_features=text_feat,
            )
            pred_sel = pred_regret.gather(dim=-1, index=selected_idx)
            targets = transform_regret_targets(
                counterfactual_regrets,
                transform=getattr(training, "regret_target_transform", "tanh"),
                valid_mask=selected_valid,
            )
            valid_float = selected_valid.float()
            loss_mse = ((pred_sel - targets).pow(2) * valid_float).sum() / valid_float.sum().clamp(min=1.0)

            if training.lambda_rank > 0.0:
                full_regret_targets = torch.zeros_like(pred_regret)
                full_regret_targets.scatter_(dim=-1, index=selected_idx, src=targets.detach())
                full_valid_mask = torch.zeros_like(base_candidate_mask, dtype=torch.bool)
                full_valid_mask.scatter_(dim=-1, index=selected_idx, src=selected_valid.bool())
                loss_rank = gap_weighted_pairwise_rank_loss(
                    scores=pred_regret,
                    regrets=full_regret_targets,
                    valid_mask=full_valid_mask,
                    margin=training.rank_margin,
                    gap_threshold=training.rank_gap_threshold,
                )
            else:
                loss_rank = pred_sel.new_tensor(0.0)
                
            loss = loss_mse + training.lambda_rank * loss_rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.grad_clip_norm > 0.0:
                grad_norm = torch.tensor(0.0, device=pred_sel.device)
                grad_norm = torch.as_tensor(
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=training.grad_clip_norm),
                    device=pred_sel.device,
                )
            optimizer.step()
            update_ema_critic_(target_critic, critic, target_ema_decay)

            step += 1
            pbar.update(1)
            if is_main_process() and step % 1 == 0:
                keep = selected_valid
                pred_regret_log = pred_regret.detach()
                pred_sel_log = pred_sel.detach()
                targets_log = targets.detach()
                diag_k = _resolve_schedule_remask_count(
                    model,
                    timesteps,
                    budget_fraction=float(training.train_remask_ratio),
                )
                regret_corr = masked_pearson_corr(pred_sel_log, counterfactual_regrets, selected_valid)
                target_corr = masked_pearson_corr(pred_sel_log, targets_log, selected_valid)
                visible_pred_mean = float(pred_regret_log[base_candidate_mask].mean().item()) if base_candidate_mask.any() else 0.0
                masked_pred_mean = float(pred_regret_log[~base_candidate_mask].mean().item()) if (~base_candidate_mask).any() else 0.0
                regret_positive_fraction = float(counterfactual_regrets[keep].gt(0.0).float().mean().item()) if keep.any() else 0.0
                pred_positive_fraction = float(pred_sel_log[keep].gt(0.0).float().mean().item()) if keep.any() else 0.0
                pred_sel_std = float(pred_sel_log[keep].std(unbiased=False).item()) if keep.any() else 0.0
                top_pred_idx, top_pred_valid = select_topk_token_positions(
                    pred_sel_log,
                    diag_k,
                    candidate_mask=selected_valid,
                    min_tokens=1,
                )
                top_pred_true = counterfactual_regrets.gather(dim=-1, index=top_pred_idx)
                top_pred_score = pred_sel_log.gather(dim=-1, index=top_pred_idx)
                top_pred_true_mean = float(top_pred_true[top_pred_valid].mean().item()) if top_pred_valid.any() else 0.0
                top_pred_score_mean = float(top_pred_score[top_pred_valid].mean().item()) if top_pred_valid.any() else 0.0
                top_pred_positive_fraction = float(top_pred_true[top_pred_valid].gt(0.0).float().mean().item()) if top_pred_valid.any() else 0.0
                oracle_idx, oracle_valid = select_topk_token_positions(
                    counterfactual_regrets,
                    diag_k,
                    candidate_mask=selected_valid,
                    min_tokens=1,
                )
                oracle_true = counterfactual_regrets.gather(dim=-1, index=oracle_idx)
                oracle_true_mean = float(oracle_true[oracle_valid].mean().item()) if oracle_valid.any() else 0.0
                tgt_mean = float(targets_log[keep].mean().item()) if keep.any() else 0.0
                tgt_std = float(targets_log[keep].std(unbiased=False).item()) if keep.any() else 0.0
                constant_target_mse = (
                    float(((targets_log - tgt_mean).pow(2) * valid_float).sum().item() / valid_float.sum().clamp(min=1.0).item())
                    if keep.any()
                    else 0.0
                )
                pbar.set_postfix({
                    "loss": float(loss.item()),
                    "rank": float(loss_rank.item()),
                    "corr": float(regret_corr.item()),
                    "t_corr": float(target_corr.item()),
                    "pos_r": regret_positive_fraction,
                    "top_r": top_pred_true_mean,
                    "reg_m": float(counterfactual_regrets[keep].mean().item()) if keep.any() else 0.0,
                })
                if tb_logger is not None and step % max(1, int(training.log_every)) == 0:
                    selected_count = float(selected_valid.float().sum().item())
                    selected_frac = float(selected_valid.float().mean().item()) if selected_valid.numel() > 0 else 0.0
                    regret_mean = float(counterfactual_regrets[keep].mean().item()) if keep.any() else 0.0
                    regret_std = float(counterfactual_regrets[keep].std(unbiased=False).item()) if keep.any() else 0.0
                    tb_logger.log_metrics(
                        step,
                        {
                            "train/loss": float(loss.item()),
                            "train/loss_mse": float(loss_mse.item()),
                            "train/loss_rank": float(loss_rank.item()),
                            "train/lambda_rank": training.lambda_rank,
                            "train/rank_margin": training.rank_margin,
                            "train/rank_gap_threshold": training.rank_gap_threshold,
                            "train/regret_corr": float(regret_corr.item()),
                            "train/target_corr": float(target_corr.item()),
                            "train/constant_target_mse": constant_target_mse,
                            "train/regret_mean": regret_mean,
                            "train/regret_std": regret_std,
                            "train/target_mean": tgt_mean,
                            "train/target_std": tgt_std,
                            "train/regret_positive_fraction": regret_positive_fraction,
                            "train/pred_positive_fraction": pred_positive_fraction,
                            "train/pred_selected_std": pred_sel_std,
                            "train/top_pred_true_regret_mean": top_pred_true_mean,
                            "train/top_pred_score_mean": top_pred_score_mean,
                            "train/top_pred_positive_fraction": top_pred_positive_fraction,
                            "train/oracle_top_true_regret_mean": oracle_true_mean,
                            "train/pred_visible_mean": visible_pred_mean,
                            "train/pred_masked_mean": masked_pred_mean,
                            "train/target_critic_ema_decay": target_ema_decay,
                            "train/use_target_critic_replay": int(use_target_critic_replay),
                            "train/fixed_rollout_step": fixed_rollout_step,
                            "train/train_repair_greedy": int(train_repair_greedy),
                            "train/train_randomize_temperature": float(training.train_randomize_temperature),
                            "train/counterfactual_rollout_steps": int(getattr(training, "counterfactual_rollout_steps", 1)),
                            "train/counterfactual_window_radius": int(getattr(training, "counterfactual_window_radius", 0)),
                            "train/counterfactual_repair_greedy": int(bool(getattr(training, "counterfactual_repair_greedy", False))),
                            "train/selected_count": selected_count,
                            "train/selected_fraction": selected_frac,
                            "train/timestep_mean": float(timesteps.mean().item()),
                        },
                    )

            if is_main_process() and step % int(training.save_every) == 0:
                critic_for_save = critic.module if isinstance(critic, DDP) else critic
                ckpt_step_path = os.path.join(experiment_cfg.output_dir, f"critic_step_{step}.pt")
                ckpt_last_path = os.path.join(experiment_cfg.output_dir, "critic_last.pt")
                save_critic_checkpoint(ckpt_step_path, critic_for_save, optimizer, step, cfg, target_critic=target_critic)
                save_critic_checkpoint(ckpt_last_path, critic_for_save, optimizer, step, cfg, target_critic=target_critic)
                if tb_logger is not None:
                    tb_logger.log_text("checkpoint/step", ckpt_step_path, step=step)
                    tb_logger.log_text("checkpoint/last", ckpt_last_path, step=step)
        if stop_step is not None and step >= stop_step:
            break
    if use_ddp:
        dist.barrier()


    pbar.close()
    if tb_logger is not None:
        tb_logger.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
