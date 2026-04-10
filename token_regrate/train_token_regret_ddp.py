import math
import os
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import open_clip

# Make project root importable when launched as scripts/train_token_regret_ddp.py via torchrun.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_VQ, get_masking_ratio, open_clip_text_encoding

from token_regrate.config import get_config
from token_regrate.train_token_regret_dataset import *
from token_regrate.utils import *


@torch.no_grad()
def forward_maskgen(
    model,
    input_ids,
    condition,
    condition_pooled,
    sample_aesthetic_score=6.5,
):
    """Forward MaskGen blocks and return logits, hidden states (x), and pooled text feature (condition_pooled).
    This is based on forward in Maskgen_VQ but adapted to return intermediate features"""
    if sample_aesthetic_score is not None:
        sample_aesthetic_score = torch.full((input_ids.shape[0],), sample_aesthetic_score, device=input_ids.device)


    embeddings = model.embeddings(input_ids)
    cond = model.text_embed_proj(condition)

    if model.micro_condition:
        condition_pooled = model.concat_micro_cond(condition_pooled, sample_aesthetic_score)
    condition_pooled = model.cond_pooled_proj(condition_pooled)

    x = embeddings + model.pos_embed[:, :embeddings.shape[1]]
    for blk in model.blocks:
        cond, x = blk(x, cond, condition_pooled.squeeze(1))
    x = model.norm(x, condition_pooled.squeeze(1))
    logits = model.lm_head(x)
    return logits, x, condition_pooled.squeeze(1)


def select_token_subset(candidate_mask, ratio_or_count, min_tokens=1):
    """Randomly select a fixed-size subset of candidate token positions."""
    bsz, seq_len = candidate_mask.shape
    k = _resolve_num_selected(seq_len, ratio_or_count, min_tokens=min_tokens)
    if k == 0:
        return (
            torch.zeros(bsz, 0, dtype=torch.long, device=candidate_mask.device),
            torch.zeros(bsz, 0, dtype=torch.bool, device=candidate_mask.device),
        )
    scores = torch.rand(bsz, seq_len, device=candidate_mask.device)
    scores = scores.masked_fill(~candidate_mask, -1.0)
    idx = scores.topk(k=k, dim=-1).indices
    valid = candidate_mask.gather(dim=-1, index=idx)
    return idx, valid


def _allocate_selection_counts(total, weights):
    """Split an integer budget across weighted buckets."""
    total = max(0, int(total))
    weights = [max(0.0, float(w)) for w in weights]
    if total == 0:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        counts = [0 for _ in weights]
        counts[0] = total
        return counts

    raw = [total * (w / weight_sum) for w in weights]
    counts = [int(math.floor(v)) for v in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def compute_logit_margin(logits):
    """Return top-1 minus top-2 logit margin per token."""
    topk = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
    if topk.shape[-1] < 2:
        return torch.zeros(logits.shape[:2], device=logits.device, dtype=logits.dtype)
    return topk[..., 0] - topk[..., 1]


def _mask_out_selected(candidate_mask, idx, valid):
    """Remove already-selected valid indices from a candidate mask."""
    if idx.numel() == 0:
        return candidate_mask
    selected_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    selected_mask.scatter_(dim=-1, index=idx, src=valid.bool())
    return candidate_mask & ~selected_mask


def select_mixed_token_subset(
    candidate_mask,
    ratio_or_count,
    logits=None,
    critic_scores=None,
    random_fraction=0.5,
    low_margin_fraction=0.25,
    critic_fraction=0.25,
    min_tokens=1,
):
    """Mix random, low-margin, and critic-prioritized token sampling for supervision."""
    bsz, seq_len = candidate_mask.shape
    total = _resolve_num_selected(seq_len, ratio_or_count, min_tokens=min_tokens)
    if total == 0:
        return (
            torch.zeros(bsz, 0, dtype=torch.long, device=candidate_mask.device),
            torch.zeros(bsz, 0, dtype=torch.bool, device=candidate_mask.device),
        )

    weights = [random_fraction, low_margin_fraction if logits is not None else 0.0, critic_fraction if critic_scores is not None else 0.0]
    random_count, low_margin_count, critic_count = _allocate_selection_counts(total, weights)

    parts_idx = []
    parts_valid = []
    remaining_mask = candidate_mask.bool()

    if random_count > 0:
        idx, valid = select_token_subset(remaining_mask, random_count, min_tokens=0)
        parts_idx.append(idx)
        parts_valid.append(valid)
        remaining_mask = _mask_out_selected(remaining_mask, idx, valid)

    if low_margin_count > 0 and logits is not None:
        low_margin_scores = -compute_logit_margin(logits)
        idx, valid = select_topk_token_positions(
            low_margin_scores,
            low_margin_count,
            candidate_mask=remaining_mask,
            min_tokens=0,
        )
        parts_idx.append(idx)
        parts_valid.append(valid)
        remaining_mask = _mask_out_selected(remaining_mask, idx, valid)

    if critic_count > 0 and critic_scores is not None:
        idx, valid = select_topk_token_positions(
            critic_scores,
            critic_count,
            candidate_mask=remaining_mask,
            min_tokens=0,
        )
        parts_idx.append(idx)
        parts_valid.append(valid)
        remaining_mask = _mask_out_selected(remaining_mask, idx, valid)

    selected_total = sum(part.shape[1] for part in parts_idx)
    if selected_total < total:
        idx, valid = select_token_subset(remaining_mask, total - selected_total, min_tokens=0)
        parts_idx.append(idx)
        parts_valid.append(valid)

    if not parts_idx:
        return (
            torch.zeros(bsz, 0, dtype=torch.long, device=candidate_mask.device),
            torch.zeros(bsz, 0, dtype=torch.bool, device=candidate_mask.device),
        )
    return torch.cat(parts_idx, dim=-1), torch.cat(parts_valid, dim=-1)


def pairwise_rank_loss(scores, utilities, valid_mask, margin=0.05):
    """Apply pairwise hinge ranking so higher utility tokens get higher scores."""
    total = scores.new_tensor(0.0)
    count = 0
    for b in range(scores.shape[0]):
        keep = valid_mask[b]
        if keep.sum() < 2:
            continue
        s = scores[b][keep]
        u = utilities[b][keep]
        higher = (u.unsqueeze(1) - u.unsqueeze(0)) > 0
        if higher.any():
            loss_mat = F.relu(float(margin) - (s.unsqueeze(1) - s.unsqueeze(0)))
            total = total + loss_mat[higher].mean()
            count += 1
    if count == 0:
        return scores.new_tensor(0.0)
    return total / count


@torch.no_grad()
def normalize_utility_per_sample(utility, valid_mask, eps=1e-4, z_clip=3.0):
    """Normalize utility values per sample on valid positions to avoid target saturation."""
    mask = valid_mask.float()
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (utility * mask).sum(dim=-1, keepdim=True) / count
    var = ((utility - mean).pow(2) * mask).sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(var + float(eps))
    z = (utility - mean) / std
    if float(z_clip) > 0.0:
        z = z.clamp(min=-float(z_clip), max=float(z_clip))
    return z


@torch.no_grad()
def build_utility_targets(counterfactual_utility, valid_mask, transform="zscore_tanh", target_scale=1.0, target_zclip=3.0):
    """
    Convert counterfactual regret values into regression targets and rank utilities.
    `zscore_tanh` is the default to prevent near-constant targets from tanh saturation.
    """
    mode = str(transform).strip().lower()
    scale = max(float(target_scale), 1e-6)

    if mode == "raw_tanh":
        rank_utility = counterfactual_utility
        targets = torch.tanh(rank_utility / scale)
        return targets, rank_utility
    if mode == "zscore_tanh":
        rank_utility = normalize_utility_per_sample(counterfactual_utility, valid_mask, z_clip=float(target_zclip))
        targets = torch.tanh(rank_utility / scale)
        return targets, rank_utility
    raise ValueError(f"Unsupported target transform: {transform}")



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
    eps = 1e-20
    noise = torch.rand_like(logits)
    g = -torch.log(-torch.log(noise.clamp(min=eps)).clamp(min=eps))
    return logits + temperature * g


@torch.no_grad()
def compute_token_nll(logits, gt_tokens):
    """Compute token-level NLL against ground-truth image tokens."""
    return F.cross_entropy(logits.transpose(1, 2), gt_tokens, reduction="none")


@torch.no_grad()
def compute_counterfactual_regret(
    model,
    z_t,
    gt_tokens,
    condition,
    condition_pooled,
    selected_idx,
    selected_valid,
    sample_aesthetic_score=6.5,
    nll_orig=None,
    counterfactual_chunk_size=64,
    neighborhood_radius=0,
):
    """Estimate token regret by explicitly remasking selected tokens and recomputing loss."""
    if nll_orig is None:
        logits_orig, _, _ = forward_maskgen(
            model,
            z_t,
            condition,
            condition_pooled,
            sample_aesthetic_score=sample_aesthetic_score,
        )
        nll_orig = compute_token_nll(logits_orig, gt_tokens)

    counterfactual_regrets = torch.zeros_like(selected_idx, dtype=nll_orig.dtype)
    pair = torch.nonzero(selected_valid, as_tuple=False)
    if pair.numel() == 0:
        return counterfactual_regrets

    seq_len = int(gt_tokens.shape[1])
    chunk_size = max(1, int(counterfactual_chunk_size))
    for start in range(0, pair.shape[0], chunk_size):
        part = pair[start:start + chunk_size]
        if part.numel() == 0:
            continue
        b_idx = part[:, 0]
        k_idx = part[:, 1]
        tok_idx = selected_idx[b_idx, k_idx]

        z_cf = z_t[b_idx].clone()
        z_cf[torch.arange(z_cf.shape[0], device=z_cf.device), tok_idx] = int(model.mask_token_id)

        logits_cf, _, _ = forward_maskgen(
            model,
            z_cf,
            condition[b_idx],
            condition_pooled[b_idx],
            sample_aesthetic_score=sample_aesthetic_score,
        )
        nll_cf_all = compute_token_nll(logits_cf, gt_tokens[b_idx])

        neigh_idx, neigh_valid = build_local_neighborhood_index(
            seq_len=seq_len,
            token_indices=tok_idx,
            radius=int(neighborhood_radius),
        )
        neigh_valid = neigh_valid.to(nll_orig.dtype)
        orig_patch = nll_orig[b_idx].gather(dim=-1, index=neigh_idx)
        cf_patch = nll_cf_all.gather(dim=-1, index=neigh_idx)

        patch_regret = (orig_patch - cf_patch) * neigh_valid
        valid_count = neigh_valid.sum(dim=-1).clamp(min=1.0)
        counterfactual_regrets[b_idx, k_idx] = patch_regret.sum(dim=-1) / valid_count

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


def schedule_linear_value(step, start_value, end_value, anneal_steps):
    """Linearly interpolate a scalar schedule from start to end."""
    start_value = float(start_value)
    end_value = float(end_value)
    anneal_steps = int(anneal_steps)
    if anneal_steps <= 0:
        return end_value
    progress = max(0.0, min(1.0, float(step) / float(anneal_steps)))
    return start_value + (end_value - start_value) * progress


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


def build_local_neighborhood_index(seq_len, token_indices, radius=1):
    """
    Build flattened neighborhood indices around each token position.
    Returns:
        neigh_idx: [M, K]
        neigh_valid: [M, K]
    """
    def infer_token_grid_shape(seq_len):
        """Infer a 2D grid shape for flattened image tokens."""
        side = int(round(math.sqrt(int(seq_len))))
        if side * side == int(seq_len):
            return side, side
        return 1, int(seq_len)
    h, w = infer_token_grid_shape(seq_len)
    m = token_indices.numel()
    device = token_indices.device

    if h == 1:
        offsets = torch.arange(-int(radius), int(radius) + 1, device=device)
        idx = token_indices[:, None] + offsets[None, :]
        valid = (idx >= 0) & (idx < int(seq_len))
        idx = idx.clamp(min=0, max=int(seq_len) - 1)
        return idx.long(), valid

    rows = torch.div(token_indices, w, rounding_mode="floor")
    cols = token_indices % w
    neigh_list = []
    valid_list = []
    for dr in range(-int(radius), int(radius) + 1):
        for dc in range(-int(radius), int(radius) + 1):
            rr = rows + dr
            cc = cols + dc
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            flat = (rr.clamp(0, h - 1) * w + cc.clamp(0, w - 1)).long()
            neigh_list.append(flat)
            valid_list.append(valid)
    neigh_idx = torch.stack(neigh_list, dim=-1)
    neigh_valid = torch.stack(valid_list, dim=-1)
    return neigh_idx, neigh_valid


@torch.no_grad()
def build_rollout_state(
    model,
    captions,
    clip_tokenizer,
    clip_encoder,
    gt_tokens,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    sample_aesthetic_score=6.5,
    remask_ratio=0.10,
    margin_threshold=0.20,
    num_sample_steps=16,
    softmax_temperature_annealing=True,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    refine_loops=1,
    refine_start_step=10,
    counterfactual_sample_ratio=0.20,
    counterfactual_chunk_size=64,
    neighborhood_radius=0,
    critic=None,
    critic_use_hidden=True,
    critic_policy_prob=0.0,
    critic_selection_noise=False,
):
    """
    Build a training state closer to inference:
    1) generate a draft with the same protocol as `generate()`
    2) remask top-k positions chosen by explicit counterfactual regret
    """
    assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]
    _ = margin_threshold  # Kept for backwards compatibility; TRC uses top-k counterfactual regret.
    draft_ids = generate_wrapper(
        model=model,
        captions=captions,
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        guidance_scale=float(guidance_scale),
        randomize_temperature=float(randomize_temperature),
        sample_aesthetic_score=float(sample_aesthetic_score),
        num_sample_steps=int(num_sample_steps),
        use_critic_head=False,
        softmax_temperature_annealing=bool(softmax_temperature_annealing),
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=float(guidance_decay_scale_pow),
        prob_sorting=bool(prob_sorting),
    ).to(gt_tokens.device)

    # Sample a rollout step from the same refinement schedule used at inference.
    loop_idx = int(torch.randint(0, refine_loops, (1,), device=gt_tokens.device).item())
    step = min(int(refine_start_step) + loop_idx, num_sample_steps - 1)
    ratio = float(step + 1) / float(num_sample_steps)
    timesteps = torch.full((gt_tokens.shape[0],), ratio, device=gt_tokens.device)

    condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
    none_cond, none_cond_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
    num_samples = condition.shape[0]
    device = condition.device
    none_cond = none_cond.repeat(num_samples, 1, 1)
    none_cond_pooled = none_cond_pooled.repeat(num_samples, 1, 1)

    cfg_scale = _compute_cfg_scale(
        ratio=ratio,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=guidance_decay_scale_pow,
        device=device,
    )

    if cfg_scale != 0.0:
        logits_all, _, _ = forward_maskgen(
            model,
            torch.cat([draft_ids, draft_ids], dim=0),
            torch.cat([condition, none_cond], dim=0),
            torch.cat([condition_pooled, none_cond_pooled], dim=0),
            sample_aesthetic_score=sample_aesthetic_score,
        )
        cond_logits, uncond_logits = logits_all[:num_samples], logits_all[num_samples:]
        logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _, _ = forward_maskgen(
            model,
            draft_ids,
            condition,
            condition_pooled,
            sample_aesthetic_score=sample_aesthetic_score,
        )

    annealed_temp = float(randomize_temperature) * (1.0 - ratio)
    if softmax_temperature_annealing:
        softmax_temperature = 0.5 + 0.8 * (1.0 - ratio)
    else:
        softmax_temperature = max(annealed_temp, 1e-6)
    logits_for_sample = logits / float(softmax_temperature)

    mask_token_id = int(model.mask_token_id)
    if 0 <= mask_token_id < logits_for_sample.shape[-1]:
        logits_for_sample[..., mask_token_id] = -1e9

    sampled_ids = _add_gumbel_noise(logits_for_sample, annealed_temp).argmax(dim=-1)
    logits_rollout, hidden_rollout, text_rollout = forward_maskgen(
        model,
        sampled_ids,
        condition,
        condition_pooled,
        sample_aesthetic_score=sample_aesthetic_score,
    )
    nll_rollout = compute_token_nll(logits_rollout, gt_tokens)

    seq_len = int(sampled_ids.shape[1])
    schedule_mask_ratio = float(get_masking_ratio(ratio, model.mask_schedule_strategy))
    if float(remask_ratio) > 0.0:
        effective_ratio = min(schedule_mask_ratio, float(remask_ratio))
    else:
        effective_ratio = schedule_mask_ratio
    num_to_mask = _resolve_num_selected(seq_len, effective_ratio, min_tokens=1)
    if num_to_mask == 0:
        return sampled_ids, timesteps

    base_candidate_mask = sampled_ids.ne(mask_token_id)
    if critic is not None and float(critic_policy_prob) > 0.0 and random.random() < float(critic_policy_prob):
        critic_scores = critic(
            hidden_states=hidden_rollout if critic_use_hidden else None,
            logits=logits_rollout,
            timesteps=timesteps,
            text_features=text_rollout,
        )
        selection_scores = critic_scores
        if bool(critic_selection_noise):
            selection_scores = _add_gumbel_noise(selection_scores, annealed_temp)
        remask_idx, remask_valid = select_topk_token_positions(
            selection_scores,
            num_to_mask,
            candidate_mask=base_candidate_mask,
            min_tokens=1,
        )
        z_t = remask_positions(sampled_ids, remask_idx, mask_token_id, valid_mask=remask_valid)
        return z_t, timesteps

    subset_size = max(
        num_to_mask,
        _resolve_num_selected(seq_len, counterfactual_sample_ratio, min_tokens=1),
    )
    selected_idx, selected_valid = select_token_subset(
        base_candidate_mask,
        subset_size,
        min_tokens=1,
    )
    counterfactual_regrets = compute_counterfactual_regret(
        model=model,
        z_t=sampled_ids,
        gt_tokens=gt_tokens,
        condition=condition,
        condition_pooled=condition_pooled,
        selected_idx=selected_idx,
        selected_valid=selected_valid,
        sample_aesthetic_score=sample_aesthetic_score,
        nll_orig=nll_rollout,
        counterfactual_chunk_size=counterfactual_chunk_size,
        neighborhood_radius=neighborhood_radius,
    )
    local_idx, local_valid = select_topk_token_positions(
        counterfactual_regrets,
        num_to_mask,
        candidate_mask=selected_valid,
        min_tokens=1,
    )
    remask_idx = selected_idx.gather(dim=-1, index=local_idx)
    remask_valid = selected_valid.gather(dim=-1, index=local_idx) & local_valid
    z_t = remask_positions(sampled_ids, remask_idx, mask_token_id, valid_mask=remask_valid)
    return z_t, timesteps

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
    margin_threshold=0.20,
    refine_start_step=10,
    critic_use_hidden=True,
    refine_softmax_temperature=0.7,
    softmax_temperature_annealing=True,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    repair_greedy=True,
):
    """Generate tokens via `model.generate()` and optionally refine from a chosen start step."""
    assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]
    assert refine_start_step >= 0, "refine_start_step must be non-negative"
    assert refine_start_step < num_sample_steps, "refine_start_step must be less than num_sample_steps"
    if bool(use_critic_head) and critic is None:
        raise ValueError("use_critic_head=True requires a non-None critic.")
    _ = margin_threshold  # Kept for backwards compatibility; TRC remasking uses top-k critic scores.


    sample_aesthetic_score = float(
        getattr(model, "sample_aesthetic_score", 6.5) if sample_aesthetic_score is None else sample_aesthetic_score
    )

    draft_ids = model.generate(
        captions=captions,
        guidance_scale=float(guidance_scale),
        randomize_temperature=float(randomize_temperature),
        sample_aesthetic_score=sample_aesthetic_score,
        softmax_temperature_annealing=bool(softmax_temperature_annealing),
        num_sample_steps=num_sample_steps,
        guidance_decay=guidance_decay,
        guidance_decay_scale_pow=float(guidance_decay_scale_pow),
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        prob_sorting=bool(prob_sorting),
    )
    if not bool(use_critic_head):
        return draft_ids

    condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
    none_cond, none_cond_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
    num_samples = condition.shape[0]
    device = condition.device
    none_cond = none_cond.repeat(num_samples, 1, 1)
    none_cond_pooled = none_cond_pooled.repeat(num_samples, 1, 1)

    ids = draft_ids.clone().to(device)
    mask_token_id = int(model.mask_token_id)


    if refine_start_step >= num_sample_steps:
        return ids

    for step in range(refine_start_step, num_sample_steps):
        ratio = float(step + 1) / float(num_sample_steps)
        ratio = float(max(1e-6, min(1.0, ratio)))
        annealed_temp = float(randomize_temperature) * (1.0 - ratio)
        is_mask = ids.eq(mask_token_id)

        cfg_scale = _compute_cfg_scale(
            ratio=ratio,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            guidance_decay_scale_pow=guidance_decay_scale_pow,
            device=device,
        )
        if cfg_scale != 0.0:
            logits_all, hidden_all, text_all = forward_maskgen(
                model,
                torch.cat([ids, ids], dim=0),
                torch.cat([condition, none_cond], dim=0),
                torch.cat([condition_pooled, none_cond_pooled], dim=0),
                sample_aesthetic_score=sample_aesthetic_score,
            )
            cond_logits, uncond_logits = logits_all[:num_samples], logits_all[num_samples:]
            logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            critic_hidden = hidden_all[:num_samples]
            critic_text = text_all[:num_samples]
        else:
            logits, critic_hidden, critic_text = forward_maskgen(
                model,
                ids,
                condition,
                condition_pooled,
                sample_aesthetic_score=sample_aesthetic_score,
            )

        timesteps = torch.full((num_samples,), ratio, device=device)
        critic_scores = critic(
            hidden_states=critic_hidden if critic_use_hidden else None,
            logits=logits,
            timesteps=timesteps,
            text_features=critic_text,
        )

        if softmax_temperature_annealing:
            softmax_temperature = 0.5 + 0.8 * (1.0 - ratio)
        else:
            softmax_temperature = max(float(refine_softmax_temperature), 1e-6)
        logits_for_sample = logits / float(softmax_temperature)

        if 0 <= mask_token_id < logits_for_sample.shape[-1]:
            logits_for_sample[..., mask_token_id] = -1e9

        if bool(repair_greedy):
            sampled_ids = logits_for_sample.argmax(dim=-1)
        else:
            sampled_ids = _add_gumbel_noise(logits_for_sample, annealed_temp).argmax(dim=-1)
        sampled_ids = torch.where(is_mask, sampled_ids, ids)

        # Final timestep keeps the sampled refinement without remasking again.
        if step == num_sample_steps - 1:
            ids = sampled_ids
            continue

        # Critic-guided repair uses deterministic top-k remasking at evaluation time.
        selection_scores = critic_scores
        select_idx, select_valid = select_topk_token_positions(
            selection_scores,
            remask_ratio,
            candidate_mask=sampled_ids.ne(mask_token_id),
        )
        ids = remask_positions(sampled_ids, select_idx, mask_token_id, valid_mask=select_valid)

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
    margin_threshold=0.20,
    refine_softmax_temperature=0.7,
    repair_greedy=True,
):
    """Generate image batch from prompts with optional critic-based token refinement."""
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
        margin_threshold=margin_threshold,
        refine_start_step=refine_start_step,
        critic_use_hidden=bool(critic_use_hidden),
        refine_softmax_temperature=refine_softmax_temperature,
        repair_greedy=bool(repair_greedy),
    ).to(model_device)
    text_guidance = prepare_text_guidance(prompts, clip_tokenizer, clip_encoder, model_device)
    image = tokenizer.decode_tokens(tokens, text_guidance)
    image = torch.clamp(image, 0.0, 1.0)
    image = (image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(arr) for arr in image]


def main():
    """Entry point for distributed token-regret critic training."""
    config = get_config()
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
    
    start_step = 0
    resume_path = str(config.runtime.resume_checkpoint).strip()
    if resume_path:
        critic_for_load = critic.module if isinstance(critic, DDP) else critic
        ckpt = load_critic_checkpoint(resume_path, critic_for_load, optimizer=optimizer, map_location=device)
        start_step = int(ckpt.get("step", 0))
        if is_main_process():
            print(f"Resumed from checkpoint: {resume_path} (step={start_step})")


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

    pbar = tqdm(total=total_steps, initial=start_step, desc="iterations", unit="iter", disable=not is_main_process())
    step = int(start_step)

    os.makedirs(experiment_cfg.output_dir, exist_ok=True)
    cfg = config.to_dict()
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
        stream = prefetch_batches(
            dataset_pipeline.iter_batches(
                batch_size=int(training.per_gpu_batch_size),
                rank=rank,
                world_size=world_size,
            ),
            prefetch_batches=stream_prefetch_batches,
        )
        for images, captions in stream:
            gt_tokens = images_to_tokens(tokenizer, images, int(model.image_seq_len)).to(device, non_blocking=True)
            train_aes = float(training.train_aesthetic_score)
            dagger_prob = schedule_linear_value(
                step=step,
                start_value=float(getattr(training, "dagger_prob_start", 0.0)),
                end_value=float(getattr(training, "dagger_prob_end", 0.0)),
                anneal_steps=int(getattr(training, "dagger_anneal_steps", 0)),
            )

            with torch.no_grad():
                # Match inference-side text conditioning (no training-time text dropout).
                condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
                condition = condition.to(device, non_blocking=True)
                condition_pooled = condition_pooled.to(device, non_blocking=True)
           
                z_t, timesteps = build_rollout_state(
                    model=model,
                    captions=captions,
                    clip_tokenizer=clip_tokenizer,
                    clip_encoder=clip_encoder,
                    gt_tokens=gt_tokens,
                    guidance_scale=float(training.train_guidance_scale),
                    randomize_temperature=float(training.train_randomize_temperature),
                    sample_aesthetic_score=float(training.train_aesthetic_score),
                    remask_ratio=float(training.train_remask_ratio),
                    margin_threshold=float(training.margin_threshold),
                    num_sample_steps=int(model_cfg.sample_steps),
                    refine_loops=int(training.refine_loops),
                    refine_start_step=int(training.train_refine_start_step),
                    counterfactual_sample_ratio=float(training.token_sample_ratio),
                    counterfactual_chunk_size=int(training.counterfactual_chunk_size),
                    neighborhood_radius=int(training.neighborhood_radius),
                    critic=critic,
                    critic_use_hidden=True,
                    critic_policy_prob=float(dagger_prob),
                    critic_selection_noise=bool(getattr(training, "dagger_selection_noise", False)),
                )

                logits_orig, hidden_orig, text_feat = forward_maskgen(
                    model,
                    z_t,
                    condition,
                    condition_pooled,
                    sample_aesthetic_score=train_aes,
                )

            pred = critic(
                hidden_states=hidden_orig,
                logits=logits_orig,
                timesteps=timesteps,
                text_features=text_feat,
            )
            base_candidate_mask = z_t.ne(int(model.mask_token_id))
            selected_idx, selected_valid = select_mixed_token_subset(
                candidate_mask=base_candidate_mask,
                ratio_or_count=float(training.token_sample_ratio),
                logits=logits_orig.detach(),
                critic_scores=pred.detach(),
                random_fraction=float(getattr(training, "training_random_fraction", 0.5)),
                low_margin_fraction=float(getattr(training, "training_low_margin_fraction", 0.25)),
                critic_fraction=float(getattr(training, "training_critic_fraction", 0.25)),
                min_tokens=1,
            )

            with torch.no_grad():
                nll_orig = compute_token_nll(logits_orig, gt_tokens)
                counterfactual_regrets = compute_counterfactual_regret(
                    model=model,
                    z_t=z_t,
                    gt_tokens=gt_tokens,
                    condition=condition,
                    condition_pooled=condition_pooled,
                    selected_idx=selected_idx,
                    selected_valid=selected_valid,
                    sample_aesthetic_score=train_aes,
                    nll_orig=nll_orig,
                    counterfactual_chunk_size=int(training.counterfactual_chunk_size),
                    neighborhood_radius=int(training.neighborhood_radius),
                )
                targets, rank_utility = build_utility_targets(
                    counterfactual_utility=counterfactual_regrets,
                    valid_mask=selected_valid,
                    transform=str(training.target_transform),
                    target_scale=float(training.target_scale),
                    target_zclip=float(training.target_zclip),
                )
            pred_sel = pred.gather(dim=-1, index=selected_idx)
            regret_corr = masked_pearson_corr(pred_sel.detach(), counterfactual_regrets, selected_valid)
            visible_pred_mean = float(pred[base_candidate_mask].mean().item()) if base_candidate_mask.any() else 0.0
            masked_pred_mean = float(pred[~base_candidate_mask].mean().item()) if (~base_candidate_mask).any() else 0.0

            denom = selected_valid.float().sum().clamp(min=1.0)
            regression_mode = str(getattr(training, "regression_loss", "huber")).strip().lower()
            if regression_mode == "huber":
                loss_reg = (
                    F.smooth_l1_loss(
                        pred_sel,
                        counterfactual_regrets,
                        reduction="none",
                        beta=float(getattr(training, "huber_beta", 0.5)),
                    ) * selected_valid.float()
                ).sum() / denom
            elif regression_mode == "mse":
                loss_reg = (((pred_sel - counterfactual_regrets) ** 2) * selected_valid.float()).sum() / denom
            else:
                raise ValueError(f"Unsupported regression_loss: {training.regression_loss}")
            loss_rank = pairwise_rank_loss(pred_sel, rank_utility, selected_valid, margin=float(training.rank_margin)) if float(training.lambda_rank) > 0.0 else torch.tensor(0.0, device=pred_sel.device)
            loss = loss_reg + float(training.lambda_rank) * loss_rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip_norm = float(getattr(training, "grad_clip_norm", 0.0))
            grad_norm = torch.tensor(0.0, device=pred_sel.device)
            if grad_clip_norm > 0.0:
                grad_norm = torch.as_tensor(
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm),
                    device=pred_sel.device,
                )
            optimizer.step()

            step += 1
            pbar.update(1)
            if is_main_process() and step % 1 == 0:
                keep = selected_valid
                tgt_mean = float(targets[keep].mean().item()) if keep.any() else 0.0
                tgt_std = float(targets[keep].std(unbiased=False).item()) if keep.any() else 0.0
                pbar.set_postfix({
                    "loss": float(loss.item()),
                    "reg": float(loss_reg.item()),
                    "rank": float(loss_rank.item()),
                    "corr": float(regret_corr.item()),
                    "reg_m": float(counterfactual_regrets[keep].mean().item()) if keep.any() else 0.0,
                    "dag": float(dagger_prob),
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
                            "train/loss_reg": float(loss_reg.item()),
                            "train/loss_rank": float(loss_rank.item()),
                            "train/regret_corr": float(regret_corr.item()),
                            "train/regret_mean": regret_mean,
                            "train/regret_std": regret_std,
                            "train/target_mean": tgt_mean,
                            "train/target_std": tgt_std,
                            "train/pred_visible_mean": visible_pred_mean,
                            "train/pred_masked_mean": masked_pred_mean,
                            "train/dagger_prob": float(dagger_prob),
                            "train/grad_norm": float(grad_norm.item()),
                            "train/selected_count": selected_count,
                            "train/selected_fraction": selected_frac,
                            "train/timestep_mean": float(timesteps.mean().item()),
                        },
                    )

            if is_main_process() and step % int(training.save_every) == 0:
                critic_for_save = critic.module if isinstance(critic, DDP) else critic
                ckpt_step_path = os.path.join(experiment_cfg.output_dir, f"critic_step_{step}.pt")
                ckpt_last_path = os.path.join(experiment_cfg.output_dir, "critic_last.pt")
                save_critic_checkpoint(ckpt_step_path, critic_for_save, optimizer, step, cfg)
                save_critic_checkpoint(ckpt_last_path, critic_for_save, optimizer, step, cfg)
                if tb_logger is not None:
                    tb_logger.log_text("checkpoint/step", ckpt_step_path, step=step)
                    tb_logger.log_text("checkpoint/last", ckpt_last_path, step=step)
    if use_ddp:
        dist.barrier()


    pbar.close()
    if tb_logger is not None:
        tb_logger.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
