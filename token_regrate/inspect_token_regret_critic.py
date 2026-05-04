import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

# Make project root importable when launched as token_regrate/inspect_token_regret_critic.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _open_jsonl(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w", encoding="utf-8")


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _tensor_stats(values):
    if values.numel() == 0:
        return {"count": 0}
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _rankdata_1d(values):
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _spearman_corr(x, y):
    if x.numel() < 2:
        return 0.0
    rx = _rankdata_1d(x.float())
    ry = _rankdata_1d(y.float())
    from token_regrate.train_token_regret_ddp import masked_pearson_corr

    return float(masked_pearson_corr(rx, ry, torch.ones_like(rx, dtype=torch.bool)).item())


def _pairwise_accuracy(scores, targets, max_pairs_per_example=20000):
    correct = 0.0
    total = 0.0
    for score, target in zip(scores, targets):
        n = int(score.numel())
        if n < 2:
            continue
        pair_count = n * (n - 1) // 2
        if pair_count <= int(max_pairs_per_example):
            i, j = torch.triu_indices(n, n, offset=1, device=score.device)
        else:
            i = torch.randint(0, n, (int(max_pairs_per_example),), device=score.device)
            j = torch.randint(0, n, (int(max_pairs_per_example),), device=score.device)
            keep = i.ne(j)
            i = i[keep]
            j = j[keep]
        target_gap = target[i] - target[j]
        useful = target_gap.ne(0)
        if not useful.any():
            continue
        pred_gap = score[i] - score[j]
        correct += float(pred_gap[useful].sign().eq(target_gap[useful].sign()).float().sum().item())
        total += float(useful.float().sum().item())
    return correct / max(total, 1.0)


def _quantile_bins(pred, target, raw_regret, bins=10):
    if pred.numel() == 0:
        return []
    order = torch.argsort(pred.float())
    chunks = torch.chunk(order, max(1, int(bins)))
    rows = []
    for bin_idx, idx in enumerate(chunks):
        if idx.numel() == 0:
            continue
        rows.append(
            {
                "bin": int(bin_idx),
                "count": int(idx.numel()),
                "pred_mean": float(pred[idx].float().mean().item()),
                "target_mean": float(target[idx].float().mean().item()),
                "raw_regret_mean": float(raw_regret[idx].float().mean().item()),
                "raw_positive_fraction": float(raw_regret[idx].gt(0.0).float().mean().item()),
            }
        )
    return rows


def _load_models(config, checkpoint, device, use_ema_target=False):
    import open_clip

    from modeling.maskgen import MaskGen_VQ
    from modeling.tatitok import TATiTok
    from token_regrate.utils import (
        initialize_token_regret_critic_model,
        load_critic_checkpoint,
        setup_cache_environment,
    )

    hf_cache_dir = setup_cache_environment()
    tok_dir = os.path.join(hf_cache_dir, "turkeyju/tokenizer_tatitok_bl128_vq")
    gen_dir = os.path.join(hf_cache_dir, "turkeyju/generator_maskgen_vq_xl")

    tokenizer = TATiTok.from_pretrained(pretrained_model_name_or_path=tok_dir, cache_dir=tok_dir).to(device)
    tokenizer.eval().requires_grad_(False)

    model = MaskGen_VQ.from_pretrained(pretrained_model_name_or_path=gen_dir, cache_dir=gen_dir).to(device)
    model.eval().requires_grad_(False)

    clip_encoder, _, _ = open_clip.create_model_and_transforms(
        str(config.model.clip_name),
        pretrained=str(config.model.clip_pretrained),
        force_quick_gelu=bool(config.model.clip_force_quick_gelu),
    )
    del clip_encoder.visual
    clip_encoder.transformer.batch_first = False
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval().requires_grad_(False)
    clip_tokenizer = open_clip.get_tokenizer(str(config.model.clip_name))

    critic = initialize_token_regret_critic_model(
        model,
        use_hidden=True,
        prompt_gap_topk=int(config.training.critic_prompt_gap_topk),
    ).to(device)
    ckpt = load_critic_checkpoint(checkpoint, critic, optimizer=None, map_location=device)
    if use_ema_target and "target_critic" in ckpt:
        target_critic = initialize_token_regret_critic_model(
            model,
            use_hidden=True,
            prompt_gap_topk=int(config.training.critic_prompt_gap_topk),
        ).to(device)
        target_critic.load_state_dict(ckpt["target_critic"], strict=True)
        critic = target_critic
    critic.eval().requires_grad_(False)
    return tokenizer, model, clip_encoder, clip_tokenizer, critic, int(ckpt.get("step", -1))


def _resolve_manifest(output_dir, manifest):
    if manifest:
        manifest_path = os.path.abspath(os.path.expanduser(str(manifest)))
        if os.path.isdir(manifest_path):
            candidates = [
                os.path.join(manifest_path, "manifest.jsonl"),
                os.path.join(manifest_path, "manifest.tsv"),
                os.path.join(manifest_path, "manifest.csv"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
            found = []
            for pattern in ("*.jsonl", "*.tsv", "*.csv"):
                found.extend(glob.glob(os.path.join(manifest_path, pattern)))
            found = sorted(os.path.abspath(path) for path in found if os.path.isfile(path))
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                names = ", ".join(os.path.basename(path) for path in found[:8])
                raise ValueError(
                    f"Manifest directory has multiple candidate files: {manifest_path} ({names}). "
                    "Pass --manifest with the exact file to use."
                )
            raise FileNotFoundError(f"No manifest file found in directory: {manifest_path}")
        return manifest_path
    return os.path.abspath(os.path.join(output_dir, "used_training_images", "manifest.jsonl"))


def _split_patterns(patterns):
    out = []
    for item in str(patterns or "").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out or ["critic_last.pt"]


def _natural_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _checkpoint_sort_key(path):
    # Keep critic_last after numbered checkpoints so time-series inspections read naturally.
    return (1 if os.path.basename(path) == "critic_last.pt" else 0, _natural_key(path))


def _find_checkpoint_files(checkpoint_dir, patterns="critic_last.pt", recursive=False):
    checkpoint_dir = os.path.abspath(os.path.expanduser(str(checkpoint_dir)))
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")

    files = []
    for pattern in _split_patterns(patterns):
        if recursive:
            search = os.path.join(checkpoint_dir, "**", pattern)
            files.extend(glob.glob(search, recursive=True))
        else:
            search = os.path.join(checkpoint_dir, pattern)
            files.extend(glob.glob(search))

    files = sorted({os.path.abspath(path) for path in files if os.path.isfile(path)}, key=_checkpoint_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir} with pattern(s): {', '.join(_split_patterns(patterns))}"
        )
    return files


def _safe_checkpoint_name(path, root=None):
    if root:
        try:
            name = os.path.relpath(path, root)
        except ValueError:
            name = os.path.basename(path)
    else:
        name = os.path.basename(path)
    stem = os.path.splitext(name)[0].replace(os.sep, "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "checkpoint"


def _iter_eval_batches(manifest, batch_size, max_examples):
    from token_regrate.dataset import TrainingDatasetPipeline

    pipeline = TrainingDatasetPipeline(mode="local", sources=manifest)
    seen = 0
    for images, captions in pipeline.iter_batches(batch_size=batch_size, rank=0, world_size=1):
        if max_examples is not None:
            remaining = int(max_examples) - seen
            if remaining <= 0:
                break
            images = images[:remaining]
            captions = captions[:remaining]
        if len(captions) == 0:
            continue
        yield images, captions, seen
        seen += len(captions)


def inspect_checkpoint(args):
    from modeling.maskgen import open_clip_text_encoding
    from token_regrate.config import get_config
    from token_regrate.train_token_regret_ddp import (
        build_rollout_state,
        compute_counterfactual_regret,
        forward_maskgen,
        masked_pearson_corr,
        select_budget_token_positions,
        select_topk_token_positions,
        transform_regret_targets,
        _resolve_refine_loop_count,
        _resolve_schedule_remask_count,
    )
    from token_regrate.utils import images_to_tokens, set_global_seed

    config = get_config()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or config.experiment.output_dir))
    config.experiment.output_dir = output_dir
    config.experiment.logging_dir = os.path.join(output_dir, "logs")
    config.runtime.resume_checkpoint = os.path.join(output_dir, "critic_last.pt")

    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint or config.runtime.resume_checkpoint))
    manifest = _resolve_manifest(output_dir, args.manifest)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(int(args.seed))
    tokenizer, model, clip_encoder, clip_tokenizer, critic, checkpoint_step = _load_models(
        config=config,
        checkpoint=checkpoint,
        device=device,
        use_ema_target=bool(args.ema_target),
    )

    training = config.training
    model_cfg = config.model
    resolved_selection = args.selection
    if resolved_selection is None:
        if bool(getattr(training, "fixed_label_cache", False)):
            resolved_selection = str(getattr(training, "label_cache_selection", "all_visible"))
        else:
            resolved_selection = "budget"
    resolved_selection = str(resolved_selection).strip().lower()
    if resolved_selection in {"visible", "all_visible"}:
        resolved_selection = "all_visible"
    elif resolved_selection == "all":
        resolved_selection = "all_visible"
    elif resolved_selection != "budget":
        raise ValueError(f"Unsupported inspection selection: {resolved_selection!r}")
    inspect_dir = os.path.abspath(os.path.expanduser(args.inspect_dir or os.path.join(output_dir, "inspection")))
    os.makedirs(inspect_dir, exist_ok=True)
    token_path = os.path.join(inspect_dir, "token_rows.jsonl")
    prompt_path = os.path.join(inspect_dir, "prompt_rows.jsonl")
    summary_path = os.path.join(inspect_dir, "summary.json")
    bins_path = os.path.join(inspect_dir, "calibration_bins.json")

    max_examples = None if int(args.max_examples) <= 0 else int(args.max_examples)
    all_pred = []
    all_target = []
    all_regret = []
    per_example_scores = []
    per_example_targets = []
    top_rows = []
    grouped = defaultdict(list)

    empty_condition, empty_condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, [""])
    empty_condition = empty_condition.to(device, non_blocking=True)
    empty_condition_pooled = empty_condition_pooled.to(device, non_blocking=True)
    decision_loop_count = _resolve_refine_loop_count(
        int(training.refine_loops),
        int(training.train_refine_start_step),
        int(model_cfg.sample_steps),
    )

    token_fp = _open_jsonl(token_path)
    prompt_fp = _open_jsonl(prompt_path)
    try:
        pbar = tqdm(
            _iter_eval_batches(manifest, int(args.batch_size), max_examples),
            total=None,
            desc="inspect",
            unit="batch",
        )
        for batch_idx, (images, captions, global_start) in enumerate(pbar):
            batch_size = len(captions)
            gt_tokens = images_to_tokens(tokenizer, images).to(device, non_blocking=True)
            with torch.no_grad():
                condition, condition_pooled = open_clip_text_encoding(clip_tokenizer, clip_encoder, captions)
                condition = condition.to(device, non_blocking=True)
                condition_pooled = condition_pooled.to(device, non_blocking=True)
                none_condition = empty_condition.expand(batch_size, -1, -1)
                none_condition_pooled = empty_condition_pooled.expand(batch_size, -1, -1)

                rollout_cycle_index = int(args.rollout_cycle_index)
                if rollout_cycle_index < 0:
                    rollout_cycle_index = batch_idx
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
                    refine_loops=int(training.refine_loops),
                    refine_start_step=int(training.train_refine_start_step),
                    repair_greedy=bool(training.train_repair_greedy),
                    fixed_step=None if int(args.fixed_step) < 0 else int(args.fixed_step),
                    rollout_step_schedule=str(args.rollout_step_schedule),
                    rollout_cycle_index=rollout_cycle_index,
                    guidance_decay="cosine",
                    guidance_decay_scale_pow=1.0,
                    prob_sorting=True,
                )

                use_prompt_gap = int(training.critic_prompt_gap_topk) > 0
                if use_prompt_gap:
                    logits_orig, hidden_orig, text_feat, cond_logits, uncond_logits = forward_maskgen(
                        model=model,
                        input_ids=z_t,
                        condition=condition,
                        condition_pooled=condition_pooled,
                        ratio=timesteps,
                        guidance_scale=float(training.train_guidance_scale),
                        sample_aesthetic_score=float(training.train_aesthetic_score),
                        guidance_decay="cosine",
                        guidance_decay_scale_pow=1.0,
                        none_condition=none_condition,
                        none_condition_pooled=none_condition_pooled,
                        return_cfg_parts=True,
                    )
                    prompt_logits_delta = cond_logits - uncond_logits if uncond_logits is not None else None
                else:
                    logits_orig, hidden_orig, text_feat = forward_maskgen(
                        model=model,
                        input_ids=z_t,
                        condition=condition,
                        condition_pooled=condition_pooled,
                        ratio=timesteps,
                        guidance_scale=float(training.train_guidance_scale),
                        sample_aesthetic_score=float(training.train_aesthetic_score),
                        guidance_decay="cosine",
                        guidance_decay_scale_pow=1.0,
                        none_condition=none_condition,
                        none_condition_pooled=none_condition_pooled,
                    )
                    prompt_logits_delta = None

                base_candidate_mask = z_t.ne(int(model.mask_token_id))
                train_label_budget = _resolve_schedule_remask_count(
                    model,
                    timesteps,
                    budget_fraction=float(training.train_remask_ratio),
                )
                if resolved_selection == "all_visible":
                    selected_idx, selected_valid = select_budget_token_positions(
                        base_candidate_mask,
                        budget_count=int(model.image_seq_len),
                        multiplier=1.0,
                        min_tokens=1,
                    )
                else:
                    selected_idx, selected_valid = select_budget_token_positions(
                        base_candidate_mask,
                        budget_count=train_label_budget,
                        multiplier=float(args.label_multiplier),
                        min_tokens=1,
                    )

                counterfactual_regrets, regret_valid = compute_counterfactual_regret(
                    model=model,
                    gt_tokens=gt_tokens,
                    z_t=z_t,
                    condition=condition,
                    condition_pooled=condition_pooled,
                    timesteps=timesteps,
                    step_indices=rollout_step_indices,
                    selected_idx=selected_idx,
                    selected_valid=selected_valid,
                    sample_aesthetic_score=float(training.train_aesthetic_score),
                    counterfactual_chunk_size=int(args.counterfactual_chunk_size or training.counterfactual_chunk_size),
                    counterfactual_rollout_steps=int(args.counterfactual_rollout_steps),
                    counterfactual_utility=str(training.counterfactual_utility),
                    counterfactual_contrast_negatives=int(training.counterfactual_contrast_negatives),
                    counterfactual_contrast_temperature=float(training.counterfactual_contrast_temperature),
                    counterfactual_contrast_mode=str(training.counterfactual_contrast_mode),
                    num_sample_steps=int(model_cfg.sample_steps),
                    guidance_scale=float(training.train_guidance_scale),
                    randomize_temperature=float(training.train_randomize_temperature),
                    guidance_decay="cosine",
                    guidance_decay_scale_pow=1.0,
                    softmax_temperature_annealing=False,
                    prob_sorting=True,
                    repair_greedy=bool(training.counterfactual_repair_greedy),
                    none_condition=none_condition,
                    none_condition_pooled=none_condition_pooled,
                )

                pred_regret = critic(
                    hidden_states=hidden_orig,
                    logits=logits_orig,
                    timesteps=timesteps,
                    text_features=text_feat,
                    prompt_logits_delta=prompt_logits_delta,
                )
                pred_sel = pred_regret.gather(dim=-1, index=selected_idx)
                targets = transform_regret_targets(
                    counterfactual_regrets,
                    transform=str(training.regret_target_transform),
                    valid_mask=regret_valid,
                )

            keep = regret_valid.bool()
            if keep.any():
                all_pred.append(pred_sel[keep].detach().cpu())
                all_target.append(targets[keep].detach().cpu())
                all_regret.append(counterfactual_regrets[keep].detach().cpu())

            diag_k = _resolve_schedule_remask_count(model, timesteps, budget_fraction=float(training.train_remask_ratio))
            top_pred_idx, top_pred_valid = select_topk_token_positions(
                pred_sel.detach(),
                diag_k,
                candidate_mask=regret_valid,
                min_tokens=1,
            )
            oracle_idx, oracle_valid = select_topk_token_positions(
                counterfactual_regrets.detach(),
                diag_k,
                candidate_mask=regret_valid,
                min_tokens=1,
            )

            for b in range(batch_size):
                valid_b = keep[b]
                if valid_b.any():
                    score_b = pred_sel[b][valid_b].detach().cpu()
                    target_b = targets[b][valid_b].detach().cpu()
                    per_example_scores.append(score_b)
                    per_example_targets.append(target_b)
                top_pred_true = counterfactual_regrets[b].gather(dim=0, index=top_pred_idx[b])
                oracle_true = counterfactual_regrets[b].gather(dim=0, index=oracle_idx[b])
                row = {
                    "global_index": int(global_start + b),
                    "batch_index": int(batch_idx),
                    "caption": captions[b],
                    "rollout_step_index": int(rollout_step_indices[b].item()),
                    "rollout_cycle_position": int(rollout_step_indices[b].item()) - int(training.train_refine_start_step),
                    "timestep": float(timesteps[b].item()),
                    "valid_count": int(valid_b.sum().item()),
                    "selected_count": int(selected_valid[b].sum().item()),
                    "visible_count": int(base_candidate_mask[b].sum().item()),
                    "top_pred_true_regret_mean": float(top_pred_true[top_pred_valid[b]].mean().item()) if top_pred_valid[b].any() else 0.0,
                    "oracle_top_true_regret_mean": float(oracle_true[oracle_valid[b]].mean().item()) if oracle_valid[b].any() else 0.0,
                    "top_pred_positions": selected_idx[b].gather(dim=0, index=top_pred_idx[b]).detach().cpu().tolist(),
                    "oracle_positions": selected_idx[b].gather(dim=0, index=oracle_idx[b]).detach().cpu().tolist(),
                }
                prompt_fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                top_rows.append(row)
                grouped[row["rollout_step_index"]].append(row)

            for b in range(batch_size):
                valid_positions = torch.nonzero(keep[b], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                for local_rank, k_idx in enumerate(valid_positions.tolist()):
                    token_row = {
                        "global_index": int(global_start + b),
                        "caption": captions[b],
                        "rollout_step_index": int(rollout_step_indices[b].item()),
                        "timestep": float(timesteps[b].item()),
                        "selected_slot": int(k_idx),
                        "token_position": int(selected_idx[b, k_idx].item()),
                        "pred": float(pred_sel[b, k_idx].item()),
                        "target": float(targets[b, k_idx].item()),
                        "raw_regret": float(counterfactual_regrets[b, k_idx].item()),
                        "pred_positive": bool(pred_sel[b, k_idx].item() > 0.0),
                        "raw_regret_positive": bool(counterfactual_regrets[b, k_idx].item() > 0.0),
                        "rank_in_valid_by_pred": int(local_rank),
                    }
                    token_fp.write(json.dumps(token_row, ensure_ascii=True) + "\n")
    finally:
        token_fp.close()
        prompt_fp.close()

    pred = torch.cat(all_pred) if all_pred else torch.empty(0)
    target = torch.cat(all_target) if all_target else torch.empty(0)
    raw_regret = torch.cat(all_regret) if all_regret else torch.empty(0)

    if pred.numel() > 0:
        mse = float((pred - target).pow(2).mean().item())
        constant_target = float((target - target.mean()).pow(2).mean().item())
        raw_constant = float((raw_regret - raw_regret.mean()).pow(2).mean().item())
        summary = {
            "checkpoint": checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "manifest": manifest,
            "inspect_dir": inspect_dir,
            "selection": resolved_selection,
            "decision_loop_count": int(decision_loop_count),
            "label_multiplier": float(args.label_multiplier),
            "num_labeled_tokens": int(pred.numel()),
            "num_prompts": int(len(top_rows)),
            "mse_vs_target": mse,
            "constant_target_mse": constant_target,
            "mse_vs_constant_ratio": mse / max(constant_target, 1e-12),
            "constant_baseline_improvement": constant_target - mse,
            "pearson_pred_target": float(masked_pearson_corr(pred, target, torch.ones_like(pred, dtype=torch.bool)).item()),
            "pearson_pred_raw_regret": float(masked_pearson_corr(pred, raw_regret, torch.ones_like(pred, dtype=torch.bool)).item()),
            "spearman_pred_target": _spearman_corr(pred, target),
            "spearman_pred_raw_regret": _spearman_corr(pred, raw_regret),
            "pairwise_accuracy_target": _pairwise_accuracy(per_example_scores, per_example_targets),
            "sign_accuracy_raw_regret": float(pred.gt(0.0).eq(raw_regret.gt(0.0)).float().mean().item()),
            "pred_positive_fraction": float(pred.gt(0.0).float().mean().item()),
            "raw_regret_positive_fraction": float(raw_regret.gt(0.0).float().mean().item()),
            "target_positive_fraction": float(target.gt(0.0).float().mean().item()),
            "pred_target_std_ratio": float(pred.std(unbiased=False).item() / max(target.std(unbiased=False).item(), 1e-12)),
            "pred_stats": _tensor_stats(pred),
            "target_stats": _tensor_stats(target),
            "raw_regret_stats": _tensor_stats(raw_regret),
            "top_pred_true_regret_mean": float(sum(r["top_pred_true_regret_mean"] for r in top_rows) / max(len(top_rows), 1)),
            "oracle_top_true_regret_mean": float(sum(r["oracle_top_true_regret_mean"] for r in top_rows) / max(len(top_rows), 1)),
        }
        summary["top_vs_oracle_ratio"] = summary["top_pred_true_regret_mean"] / max(summary["oracle_top_true_regret_mean"], 1e-12)
    else:
        summary = {
            "checkpoint": checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "manifest": manifest,
            "inspect_dir": inspect_dir,
            "num_labeled_tokens": 0,
            "error": "No valid counterfactual labels were produced.",
        }

    by_step = {}
    for step, rows in grouped.items():
        by_step[str(step)] = {
            "num_prompts": int(len(rows)),
            "top_pred_true_regret_mean": float(sum(r["top_pred_true_regret_mean"] for r in rows) / max(len(rows), 1)),
            "oracle_top_true_regret_mean": float(sum(r["oracle_top_true_regret_mean"] for r in rows) / max(len(rows), 1)),
        }
        by_step[str(step)]["top_vs_oracle_ratio"] = (
            by_step[str(step)]["top_pred_true_regret_mean"]
            / max(by_step[str(step)]["oracle_top_true_regret_mean"], 1e-12)
        )
    summary["by_rollout_step"] = by_step

    bins = _quantile_bins(pred, target, raw_regret, bins=int(args.bins))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(bins_path, "w", encoding="utf-8") as f:
        json.dump(bins, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote token rows: {token_path}")
    print(f"Wrote prompt rows: {prompt_path}")
    print(f"Wrote calibration bins: {bins_path}")
    print(f"Wrote summary: {summary_path}")
    return summary


def inspect_directory(args):
    checkpoint_dir = os.path.abspath(os.path.expanduser(args.directory))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or checkpoint_dir))
    inspect_root = os.path.abspath(os.path.expanduser(args.inspect_root or os.path.join(output_dir, "inspection_all")))
    os.makedirs(inspect_root, exist_ok=True)

    checkpoints = _find_checkpoint_files(
        checkpoint_dir=checkpoint_dir,
        patterns=args.checkpoint_pattern,
        recursive=bool(args.recursive),
    )

    summaries = []
    for checkpoint in checkpoints:
        checkpoint_name = _safe_checkpoint_name(checkpoint, root=checkpoint_dir)
        inspect_dir = os.path.join(inspect_root, checkpoint_name)
        summary_path = os.path.join(inspect_dir, "summary.json")
        if bool(args.skip_existing) and os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summaries.append(summary)
            print(f"Skipping existing inspection: {summary_path}")
            continue

        child_args = argparse.Namespace(**vars(args))
        child_args.output_dir = output_dir
        child_args.checkpoint = checkpoint
        child_args.inspect_dir = inspect_dir
        summary = inspect_checkpoint(child_args)
        summaries.append(summary)

    aggregate = {
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "inspect_root": inspect_root,
        "checkpoint_pattern": args.checkpoint_pattern,
        "recursive": bool(args.recursive),
        "num_checkpoints": int(len(checkpoints)),
        "summaries": summaries,
    }
    aggregate_path = os.path.join(inspect_root, "directory_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(f"Wrote directory summary: {aggregate_path}")


def _none_if_nonpositive(value):
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def _resolve_geneval_metadata(geneval_root, metadata):
    if metadata:
        path = os.path.abspath(os.path.expanduser(str(metadata)))
    else:
        path = os.path.join(os.path.abspath(os.path.expanduser(str(geneval_root))), "prompts", "evaluation_metadata.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GenEval metadata not found: {path}")
    return path


def _count_geneval_metadata(path, max_prompts=None):
    rows = 0
    tags = defaultdict(int)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            tags[str(row.get("tag", ""))] += 1
            if max_prompts is not None and rows >= int(max_prompts):
                break
    return rows, dict(sorted(tags.items()))


def _write_balanced_geneval_metadata(source_path, out_path, prompts_per_task=0, max_prompts=None):
    """Write a GenEval metadata subset, optionally capped per task tag."""
    prompts_per_task = int(prompts_per_task or 0)
    max_prompts = _none_if_nonpositive(max_prompts)
    if prompts_per_task <= 0 and max_prompts is None:
        return source_path

    rows = []
    tag_counts = defaultdict(int)
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tag = str(row.get("tag", ""))
            if prompts_per_task > 0 and tag_counts[tag] >= prompts_per_task:
                continue
            rows.append(row)
            tag_counts[tag] += 1
            if max_prompts is not None and len(rows) >= int(max_prompts):
                break

    if not rows:
        raise RuntimeError(f"No GenEval metadata rows selected from: {source_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out_path


def _has_geneval_images(image_dir, min_prompts=1, min_samples_per_prompt=1):
    if not os.path.isdir(image_dir):
        return False
    ready = 0
    for name in os.listdir(image_dir):
        if not name.isdigit():
            continue
        prompt_dir = os.path.join(image_dir, name)
        samples_dir = os.path.join(prompt_dir, "samples")
        metadata_path = os.path.join(prompt_dir, "metadata.jsonl")
        if not os.path.isfile(metadata_path) or not os.path.isdir(samples_dir):
            continue
        sample_files = [
            file_name
            for file_name in os.listdir(samples_dir)
            if re.match(r"\d+\.png$", file_name)
        ]
        if len(sample_files) >= int(min_samples_per_prompt):
            ready += 1
    return ready >= int(min_prompts)


def _run_official_geneval_eval(args, image_dir, outfile):
    geneval_root = os.path.abspath(os.path.expanduser(str(args.geneval_root)))
    eval_script = os.path.join(geneval_root, "evaluation", "evaluate_images.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"GenEval evaluator not found: {eval_script}")

    cmd = [
        os.path.abspath(os.path.expanduser(str(args.eval_python or sys.executable))),
        eval_script,
        os.path.abspath(image_dir),
        "--outfile",
        os.path.abspath(outfile),
        "--model-path",
        os.path.abspath(os.path.expanduser(str(args.geneval_model_path))),
    ]
    if args.geneval_model_config:
        cmd.extend(["--model-config", os.path.abspath(os.path.expanduser(str(args.geneval_model_config)))])
    if args.geneval_eval_option:
        cmd.append("--options")
        cmd.extend(str(item) for item in args.geneval_eval_option)

    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    stderr_path = os.path.splitext(os.path.abspath(outfile))[0] + ".stderr.txt"
    stdout_path = os.path.splitext(os.path.abspath(outfile))[0] + ".stdout.txt"
    with open(stdout_path, "w", encoding="utf-8") as stdout_fp, open(stderr_path, "w", encoding="utf-8") as stderr_fp:
        proc = subprocess.run(cmd, stdout=stdout_fp, stderr=stderr_fp, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Official GenEval evaluation failed for {image_dir}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return outfile


def _extract_geneval_sample_index(filename):
    match = re.search(r"/(\d{5})/samples/(\d{4})\.png$", str(filename).replace(os.sep, "/"))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _summarize_geneval_results(path):
    rows = list(_read_jsonl(path))
    if not rows:
        raise RuntimeError(f"No GenEval rows found: {path}")

    by_tag = defaultdict(list)
    prompt_success = defaultdict(list)
    sample_rows = {}
    for row in rows:
        correct = bool(row.get("correct", False))
        tag = str(row.get("tag", ""))
        metadata = str(row.get("metadata", ""))
        prompt_success[metadata].append(correct)
        by_tag[tag].append(correct)
        prompt_idx, sample_idx = _extract_geneval_sample_index(row.get("filename", ""))
        key = (metadata, prompt_idx, sample_idx)
        sample_rows[key] = {
            "correct": correct,
            "tag": tag,
            "prompt": row.get("prompt", ""),
            "reason": row.get("reason", ""),
            "filename": row.get("filename", ""),
            "prompt_index": prompt_idx,
            "sample_index": sample_idx,
        }

    tag_scores = {
        tag: {
            "num_images": int(len(values)),
            "num_correct": int(sum(1 for item in values if item)),
            "score": float(sum(1 for item in values if item) / max(len(values), 1)),
        }
        for tag, values in by_tag.items()
    }
    prompt_values = [any(values) for values in prompt_success.values()]
    return {
        "results_path": os.path.abspath(path),
        "num_images": int(len(rows)),
        "num_prompts": int(len(prompt_success)),
        "image_correct_fraction": float(sum(1 for row in rows if bool(row.get("correct", False))) / max(len(rows), 1)),
        "prompt_correct_fraction": float(sum(1 for item in prompt_values if item) / max(len(prompt_values), 1)),
        "overall_score": float(sum(item["score"] for item in tag_scores.values()) / max(len(tag_scores), 1)),
        "by_tag": tag_scores,
        "_prompt_success": {key: bool(any(values)) for key, values in prompt_success.items()},
        "_sample_rows": sample_rows,
    }


def _public_geneval_summary(summary):
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def _compare_geneval_summaries(baseline, head):
    all_tags = sorted(set(baseline["by_tag"]) | set(head["by_tag"]))
    by_tag = {}
    for tag in all_tags:
        baseline_score = float(baseline["by_tag"].get(tag, {}).get("score", 0.0))
        head_score = float(head["by_tag"].get(tag, {}).get("score", 0.0))
        by_tag[tag] = {
            "baseline": baseline_score,
            "head": head_score,
            "delta": head_score - baseline_score,
        }

    baseline_prompts = baseline["_prompt_success"]
    head_prompts = head["_prompt_success"]
    shared_prompts = sorted(set(baseline_prompts) & set(head_prompts))
    prompt_improvements = sum(1 for key in shared_prompts if not baseline_prompts[key] and head_prompts[key])
    prompt_regressions = sum(1 for key in shared_prompts if baseline_prompts[key] and not head_prompts[key])

    baseline_samples = baseline["_sample_rows"]
    head_samples = head["_sample_rows"]
    shared_samples = sorted(set(baseline_samples) & set(head_samples))
    sample_improvements = sum(
        1 for key in shared_samples if not baseline_samples[key]["correct"] and head_samples[key]["correct"]
    )
    sample_regressions = sum(
        1 for key in shared_samples if baseline_samples[key]["correct"] and not head_samples[key]["correct"]
    )

    return {
        "baseline_overall_score": float(baseline["overall_score"]),
        "head_overall_score": float(head["overall_score"]),
        "overall_score_delta": float(head["overall_score"] - baseline["overall_score"]),
        "baseline_image_correct_fraction": float(baseline["image_correct_fraction"]),
        "head_image_correct_fraction": float(head["image_correct_fraction"]),
        "image_correct_fraction_delta": float(head["image_correct_fraction"] - baseline["image_correct_fraction"]),
        "baseline_prompt_correct_fraction": float(baseline["prompt_correct_fraction"]),
        "head_prompt_correct_fraction": float(head["prompt_correct_fraction"]),
        "prompt_correct_fraction_delta": float(head["prompt_correct_fraction"] - baseline["prompt_correct_fraction"]),
        "prompt_improvements": int(prompt_improvements),
        "prompt_regressions": int(prompt_regressions),
        "prompt_net_improvements": int(prompt_improvements - prompt_regressions),
        "sample_improvements": int(sample_improvements),
        "sample_regressions": int(sample_regressions),
        "sample_net_improvements": int(sample_improvements - sample_regressions),
        "by_tag": by_tag,
    }


def inspect_geneval(args):
    from token_regrate.config import get_config
    from token_regrate.utils import set_global_seed
    from geneval_parallel import generate_geneval_images_maskgen

    config = get_config()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or config.experiment.output_dir))
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint or os.path.join(output_dir, "critic_last.pt")))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    checkpoint_name = _safe_checkpoint_name(checkpoint, root=output_dir)
    geneval_dir = os.path.abspath(
        os.path.expanduser(args.geneval_dir or os.path.join(output_dir, "geneval_inspection", checkpoint_name))
    )
    image_root = os.path.join(geneval_dir, "images")
    result_root = os.path.join(geneval_dir, "results")
    baseline_dir = os.path.join(image_root, "baseline")
    head_dir = os.path.join(image_root, "head")
    baseline_results = os.path.join(result_root, "baseline_results.jsonl")
    head_results = os.path.join(result_root, "head_results.jsonl")
    report_path = os.path.join(geneval_dir, "comparison_summary.json")
    os.makedirs(result_root, exist_ok=True)

    source_metadata_path = _resolve_geneval_metadata(args.geneval_root, args.metadata)
    metadata_path = _write_balanced_geneval_metadata(
        source_path=source_metadata_path,
        out_path=os.path.join(geneval_dir, "selected_metadata.jsonl"),
        prompts_per_task=int(args.prompts_per_task),
        max_prompts=args.max_prompts,
    )
    prompt_count, prompt_tags = _count_geneval_metadata(metadata_path)
    if prompt_count <= 0:
        raise RuntimeError(f"No GenEval prompts found: {metadata_path}")
    max_prompts = None

    set_global_seed(int(args.seed))
    if not bool(args.skip_generation):
        baseline_ready = bool(args.skip_existing) and _has_geneval_images(
            baseline_dir,
            min_prompts=prompt_count,
            min_samples_per_prompt=int(args.batch_size),
        )
        head_ready = bool(args.skip_existing) and _has_geneval_images(
            head_dir,
            min_prompts=prompt_count,
            min_samples_per_prompt=int(args.batch_size),
        )
        if not baseline_ready or not head_ready:
            device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            tokenizer, model, clip_encoder, clip_tokenizer, critic, checkpoint_step = _load_models(
                config=config,
                checkpoint=checkpoint,
                device=device,
                use_ema_target=bool(args.ema_target),
            )
            training = config.training
            model_cfg = config.model
            if not baseline_ready:
                generate_geneval_images_maskgen(
                    model=model,
                    tokenizer=tokenizer,
                    clip_tokenizer=clip_tokenizer,
                    clip_encoder=clip_encoder,
                    metadata_jsonl=metadata_path,
                    outdir=baseline_dir,
                    batch_size=int(args.batch_size),
                    max_prompts=max_prompts,
                    seed=int(args.seed),
                    guidance_scale=float(args.guidance_scale),
                    randomize_temperature=float(args.randomize_temperature),
                    aesthetic_score=float(args.aesthetic_score),
                    num_sample_steps=int(args.num_sample_steps or model_cfg.sample_steps),
                    use_regret_remask=False,
                    critic=None,
                    remask_ratio=float(args.remask_ratio or training.train_remask_ratio),
                    refine_start_step=int(args.refine_start_step if args.refine_start_step >= 0 else training.train_refine_start_step),
                    refine_loops=_none_if_nonpositive(args.refine_loops) or int(training.refine_loops),
                    critic_use_hidden=True,
                    repair_greedy=bool(args.repair_greedy),
                    attention_backend=str(args.attention_backend),
                    clip_force_quick_gelu=bool(model_cfg.clip_force_quick_gelu),
                )
            else:
                print(f"Reusing existing baseline images: {baseline_dir}")

            if not head_ready:
                generate_geneval_images_maskgen(
                    model=model,
                    tokenizer=tokenizer,
                    clip_tokenizer=clip_tokenizer,
                    clip_encoder=clip_encoder,
                    metadata_jsonl=metadata_path,
                    outdir=head_dir,
                    batch_size=int(args.batch_size),
                    max_prompts=max_prompts,
                    seed=int(args.seed),
                    guidance_scale=float(args.guidance_scale),
                    randomize_temperature=float(args.randomize_temperature),
                    aesthetic_score=float(args.aesthetic_score),
                    num_sample_steps=int(args.num_sample_steps or model_cfg.sample_steps),
                    use_regret_remask=True,
                    critic=critic,
                    remask_ratio=float(args.remask_ratio or training.train_remask_ratio),
                    refine_start_step=int(args.refine_start_step if args.refine_start_step >= 0 else training.train_refine_start_step),
                    refine_loops=_none_if_nonpositive(args.refine_loops) or int(training.refine_loops),
                    critic_use_hidden=True,
                    repair_greedy=bool(args.repair_greedy),
                    attention_backend=str(args.attention_backend),
                    clip_force_quick_gelu=bool(model_cfg.clip_force_quick_gelu),
                )
            else:
                print(f"Reusing existing head images: {head_dir}")
        else:
            checkpoint_step = -1
            print(f"Reusing existing GenEval images in: {image_root}")
    else:
        checkpoint_step = -1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not bool(args.skip_eval):
        if not (bool(args.skip_existing) and os.path.isfile(baseline_results)):
            _run_official_geneval_eval(args, baseline_dir, baseline_results)
        else:
            print(f"Reusing existing baseline results: {baseline_results}")
        if not (bool(args.skip_existing) and os.path.isfile(head_results)):
            _run_official_geneval_eval(args, head_dir, head_results)
        else:
            print(f"Reusing existing head results: {head_results}")

    if not os.path.isfile(baseline_results) or not os.path.isfile(head_results):
        report = {
            "checkpoint": checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "metadata": metadata_path,
            "source_metadata": source_metadata_path,
            "geneval_dir": geneval_dir,
            "baseline_images": baseline_dir,
            "head_images": head_dir,
            "baseline_results": baseline_results,
            "head_results": head_results,
            "num_prompts": int(prompt_count),
            "prompt_tags": prompt_tags,
            "status": "generated_images_only",
        }
    else:
        baseline_summary = _summarize_geneval_results(baseline_results)
        head_summary = _summarize_geneval_results(head_results)
        report = {
            "checkpoint": checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "metadata": metadata_path,
            "source_metadata": source_metadata_path,
            "geneval_dir": geneval_dir,
            "baseline_images": baseline_dir,
            "head_images": head_dir,
            "baseline": _public_geneval_summary(baseline_summary),
            "head": _public_geneval_summary(head_summary),
            "comparison": _compare_geneval_summaries(baseline_summary, head_summary),
            "num_prompts": int(prompt_count),
            "prompt_tags": prompt_tags,
            "status": "evaluated",
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote GenEval comparison summary: {report_path}")
    return report


def summarize_logs(args):
    metrics_path = os.path.abspath(os.path.expanduser(args.metrics))
    rows = list(_read_jsonl(metrics_path))
    if not rows:
        raise RuntimeError(f"No metrics rows found: {metrics_path}")
    keys = [
        "train/loss",
        "train/loss_mse",
        "train/regret_corr",
        "train/target_corr",
        "train/loss_mse_vs_constant_ratio",
        "train/constant_baseline_improvement",
        "train/pred_target_std_ratio",
        "train/pred_positive_fraction",
        "train/top_pred_true_regret_mean",
        "train/oracle_top_true_regret_mean",
        "train/regret_valid_fraction",
    ]
    window = max(1, int(args.window))
    segments = {
        "first": rows[:window],
        "last": rows[-window:],
    }
    if len(rows) > window * 2:
        mid = len(rows) // 2
        segments["middle"] = rows[max(0, mid - window // 2): min(len(rows), mid + math.ceil(window / 2))]

    summary = {
        "metrics": metrics_path,
        "num_rows": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "window": int(window),
        "segments": {},
        "by_rollout_step": {},
    }
    for name, seg in segments.items():
        block = {"first_step": int(seg[0]["step"]), "last_step": int(seg[-1]["step"])}
        for key in keys:
            vals = [float(row["metrics"][key]) for row in seg if key in row["metrics"]]
            if vals:
                t = torch.tensor(vals)
                block[key] = _tensor_stats(t)
        summary["segments"][name] = block

    grouped = defaultdict(list)
    for row in rows:
        step_idx = row["metrics"].get("train/rollout_step_index")
        if step_idx is not None:
            grouped[int(step_idx)].append(row)
    for step_idx, seg in sorted(grouped.items()):
        block = {"count": int(len(seg))}
        for key in keys:
            vals = [float(row["metrics"][key]) for row in seg if key in row["metrics"]]
            if vals:
                block[key] = _tensor_stats(torch.tensor(vals))
        summary["by_rollout_step"][str(step_idx)] = block

    out_path = args.output
    if out_path:
        out_path = os.path.abspath(os.path.expanduser(out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote log summary: {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect token-regret critic training logs or a checkpoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    logs = sub.add_parser("logs", help="Summarize metrics.jsonl without loading models.")
    logs.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    logs.add_argument("--window", type=int, default=20)
    logs.add_argument("--output", required=True, help="Path to write summary.json")
    logs.set_defaults(func=summarize_logs)

    ckpt = sub.add_parser("checkpoint", help="Inspect a critic checkpoint by running inference on a sample of training data.")
    ckpt.add_argument("--output-dir", required=True, help="Base output directory to resolve defaults for other paths.")
    ckpt.add_argument("--checkpoint", required=True, help="Path to critic checkpoint to inspect.")
    ckpt.add_argument("--manifest", default="")
    ckpt.add_argument("--inspect-dir", default="")
    ckpt.add_argument("--device", default="")
    ckpt.add_argument("--batch-size", type=int, default=8)
    ckpt.add_argument("--max-examples", type=int, default=32)
    ckpt.add_argument("--seed", type=int, default=123)
    ckpt.add_argument("--selection", choices=("budget", "all", "all_visible", "visible"), default=None)
    ckpt.add_argument("--label-multiplier", type=float, default=2.0)
    ckpt.add_argument("--counterfactual-rollout-steps", type=int, default=2)
    ckpt.add_argument("--counterfactual-chunk-size", type=int, default=0)
    ckpt.add_argument("--fixed-step", type=int, default=-1)
    ckpt.add_argument("--rollout-step-schedule", choices=("cycle", "random"), default="cycle")
    ckpt.add_argument("--rollout-cycle-index", type=int, default=-1)
    ckpt.add_argument("--ema-target", action="store_true")
    ckpt.add_argument("--bins", type=int, default=10)
    ckpt.set_defaults(func=inspect_checkpoint)

    directory = sub.add_parser(
        "directory",
        help="Inspect every matching critic checkpoint in a directory and save one inspection folder per checkpoint.",
    )
    directory.add_argument("directory", help="Directory containing critic checkpoint files.")
    directory.add_argument(
        "--output-dir",
        default="",
        help="Base output directory for resolving defaults. Defaults to the checkpoint directory.",
    )
    directory.add_argument(
        "--checkpoint-pattern",
        default="critic_last.pt",
        help="Comma-separated glob pattern(s) to match checkpoint files. Default: critic_last.pt",
    )
    directory.add_argument("--recursive", action="store_true", help="Search checkpoint files recursively.")
    directory.add_argument(
        "--manifest",
        default="",
        help="Manifest file or directory. Directory mode looks for manifest.jsonl/tsv/csv automatically.",
    )
    directory.add_argument(
        "--inspect-root",
        default="",
        help="Directory where per-checkpoint inspection folders are written. Default: output_dir/inspection_all.",
    )
    directory.add_argument("--device", default="")
    directory.add_argument("--batch-size", type=int, default=8)
    directory.add_argument("--max-examples", type=int, default=32)
    directory.add_argument("--seed", type=int, default=123)
    directory.add_argument("--selection", choices=("budget", "all", "all_visible", "visible"), default=None)
    directory.add_argument("--label-multiplier", type=float, default=2.0)
    directory.add_argument("--counterfactual-rollout-steps", type=int, default=2)
    directory.add_argument("--counterfactual-chunk-size", type=int, default=0)
    directory.add_argument("--fixed-step", type=int, default=-1)
    directory.add_argument("--rollout-step-schedule", choices=("cycle", "random"), default="cycle")
    directory.add_argument("--rollout-cycle-index", type=int, default=-1)
    directory.add_argument("--ema-target", action="store_true")
    directory.add_argument("--bins", type=int, default=10)
    directory.add_argument("--skip-existing", action="store_true")
    directory.set_defaults(func=inspect_directory)

    geneval = sub.add_parser(
        "geneval",
        help="Run a GenEval baseline-vs-critic comparison for a checkpoint.",
    )
    geneval.add_argument("--output-dir", required=True, help="Base output directory containing critic checkpoints.")
    geneval.add_argument("--checkpoint", required=True, help="Path to critic checkpoint to compare against baseline.")
    geneval.add_argument(
        "--geneval-root",
        default="geneval",
        help="Path to the GenEval repository/assets. Default: geneval",
    )
    geneval.add_argument(
        "--metadata",
        default="",
        help="GenEval-style metadata JSONL. Defaults to geneval/prompts/evaluation_metadata.jsonl.",
    )
    geneval.add_argument(
        "--geneval-dir",
        default="",
        help="Output directory for generated images, evaluator results, and comparison summary.",
    )
    geneval.add_argument("--device", default="")
    geneval.add_argument("--batch-size", type=int, default=4, help="Number of generated samples per prompt.")
    geneval.add_argument("--max-prompts", type=int, default=0, help="Limit prompts for a quick smoke test; <=0 uses all.")
    geneval.add_argument(
        "--prompts-per-task",
        type=int,
        default=4,
        help="Select this many prompts from each GenEval task tag; <=0 disables balanced selection.",
    )
    geneval.add_argument("--seed", type=int, default=42)
    geneval.add_argument("--guidance-scale", type=float, default=12.0)
    geneval.add_argument("--randomize-temperature", type=float, default=2.0)
    geneval.add_argument("--aesthetic-score", type=float, default=6.5)
    geneval.add_argument("--num-sample-steps", type=int, default=0, help="<=0 uses config.model.sample_steps.")
    geneval.add_argument("--remask-ratio", type=float, default=0.0, help="<=0 uses config.training.train_remask_ratio.")
    geneval.add_argument("--refine-start-step", type=int, default=-1, help="<0 uses config.training.train_refine_start_step.")
    geneval.add_argument("--refine-loops", type=int, default=0, help="<=0 uses config.training.refine_loops.")
    geneval.add_argument("--repair-greedy", action="store_true")
    geneval.add_argument("--ema-target", action="store_true")
    geneval.add_argument("--skip-existing", action="store_true", help="Reuse complete images/results when present.")
    geneval.add_argument("--skip-generation", action="store_true", help="Only evaluate/compare existing generated images.")
    geneval.add_argument("--skip-eval", action="store_true", help="Only generate images; do not run official GenEval.")
    geneval.add_argument(
        "--eval-python",
        default=sys.executable,
        help="Python executable for GenEval evaluation, e.g. a geneval conda env.",
    )
    geneval.add_argument(
        "--geneval-model-path",
        default=os.path.join("geneval", "saved_models"),
        help="Directory containing the GenEval detector checkpoint.",
    )
    geneval.add_argument(
        "--geneval-model-config",
        default="",
        help="Optional detector config passed to GenEval evaluate_images.py.",
    )
    geneval.add_argument(
        "--geneval-eval-option",
        action="append",
        default=[],
        help="Extra key=value option passed after evaluate_images.py --options. May be repeated.",
    )
    geneval.add_argument(
        "--attention-backend",
        choices=("math", "flash", "xformers"),
        default=os.environ.get("MASKGEN_ATTENTION_MODE", "math"),
    )
    geneval.set_defaults(func=inspect_geneval)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()





# python token_regrate/inspect_token_regret_critic.py directory \
#   outputs/token_regret_critic_tanh_ngloss \
#   --manifest outputs/token_regret_critic_tanh_ngloss/used_training_images \
#   --batch-size 8 \
#   --max-examples 32 \
#   --skip-existing


# python token_regrate/inspect_token_regret_critic.py directory \
#   outputs/token_regret_critic_tanh_ngloss_1 \
#   --manifest outputs/token_regret_critic_tanh_ngloss_1/used_training_images \
#   --batch-size 8 \
#   --max-examples 32 


# python token_regrate/inspect_token_regret_critic.py geneval \
#   --output-dir outputs/token_regret_critic_tanh_ngloss_prompt_cach \
#   --checkpoint outputs/token_regret_critic_tanh_ngloss_prompt_cach/critic_last.pt \
#   --batch-size 4 \
#   --prompts-per-task 4 \
#   --skip-existing \
#   --eval-python /home/behzad/anaconda3/envs/geneval/bin/python


# python token_regrate/inspect_token_regret_critic.py geneval \
#   --output-dir outputs/token_regret_critic_tanh_ngloss_prompt_cach \
#   --checkpoint outputs/token_regret_critic_tanh_ngloss_prompt_cach/critic_last.pt \
#   --batch-size 4 \
#   --prompts-per-task 4 \
#   --skip-existing \
#   --eval-python /home/behzad/anaconda3/envs/geneval/bin/python
