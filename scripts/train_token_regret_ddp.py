import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import open_clip

try:
    import webdataset as wds
except Exception:
    wds = None

# Make project root importable when launched as scripts/train_token_regret_ddp.py via torchrun.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_VQ, get_masking_ratio


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or is_dist():
        return
    dist.init_process_group(backend="gloo", init_method="env://")


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_hf_shard_urls(num_shards=69):
    base_url = (
        "https://huggingface.co/datasets/ProGamerGov/"
        "synthetic-dataset-1m-high-quality-captions/"
        "resolve/main/data/data-{i:06d}.tar"
    )
    return [base_url.format(i=i) for i in range(int(num_shards))]


def _default_cc12m_tsv_path():
    return os.path.join("dataset", "cc12m.tsv")


def _default_cc12m_cache_dir():
    return os.path.join("dataset", "cc12m_image_cache")


def _is_local_cc12m_tsv(source):
    if source is None:
        return False
    source_str = str(source).strip()
    return bool(source_str) and os.path.isfile(source_str) and source_str.lower().endswith((".tsv", ".csv"))


def _normalize_dataset_source(source):
    if source is None:
        return {"kind": "hf_webdataset", "value": []}
    if isinstance(source, (list, tuple)):
        items = [str(x).strip() for x in source if str(x).strip()]
        if len(items) == 1 and _is_local_cc12m_tsv(items[0]):
            return {"kind": "cc12m_tsv", "value": items[0]}
        return {"kind": "hf_webdataset", "value": items}
    source_str = str(source).strip()
    if _is_local_cc12m_tsv(source_str):
        return {"kind": "cc12m_tsv", "value": source_str}
    return {"kind": "hf_webdataset", "value": [source_str] if source_str else []}


def _count_valid_cc12m_rows(tsv_path):
    total = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            image_url, caption = line.split("\t", 1)
            if image_url.strip() and caption.strip():
                total += 1
    return total


def _infer_total_examples_from_urls(urls):
    source = _normalize_dataset_source(urls)
    if source["kind"] == "cc12m_tsv":
        try:
            return _count_valid_cc12m_rows(source["value"])
        except Exception:
            return None

    if urls is None or len(urls) == 0:
        return None

    try:
        default_urls = _default_hf_shard_urls()
        if len(urls) == len(default_urls) and all(str(a) == str(b) for a, b in zip(urls, default_urls)):
            return 1_000_000
    except Exception:
        pass
    return None


def _extract_caption_from_sample(sample):
    for key in ("txt", "caption", "text", "prompt"):
        val = sample.get(key)
        if isinstance(val, str) and val.strip() != "":
            return val.strip()

    meta = sample.get("json")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if isinstance(meta, dict):
        for key in ("short_caption", "long_caption", "caption", "text", "prompt"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip() != "":
                return val.strip()
    return None


def _extract_pil_image_from_sample(sample):
    for key in ("jpg", "jpeg", "png", "webp", "image"):
        if key not in sample:
            continue
        obj = sample[key]
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, dict) and "bytes" in obj and obj["bytes"] is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    return None


def _ensure_cc12m_cache_dir(cache_dir):
    cache_dir = str(cache_dir).strip() if cache_dir is not None else ""
    if not cache_dir:
        cache_dir = _default_cc12m_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cc12m_cache_path(image_url, cache_dir):
    parsed = urlparse(str(image_url))
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        ext = ".img"
    key = hashlib.sha1(str(image_url).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, key + ext)


def _fetch_cc12m_image(image_url, timeout=30, retries=3, cache_dir=None, use_cache=True):
    cache_path = None
    if bool(use_cache):
        cache_dir = _ensure_cc12m_cache_dir(cache_dir)
        cache_path = _cc12m_cache_path(image_url, cache_dir)
        if os.path.isfile(cache_path):
            try:
                with Image.open(cache_path) as img:
                    return img.convert("RGB")
            except Exception:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

    last_error = None
    request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(max(1, int(retries))):
        try:
            with urlopen(request, timeout=float(timeout)) as response:
                image_bytes = response.read()
            if cache_path is not None:
                try:
                    tmp_path = cache_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(image_bytes)
                    os.replace(tmp_path, cache_path)
                except Exception:
                    pass
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")
        except Exception as exc:
            last_error = exc
    raise last_error


def _center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _tokenizer_image_size(tokenizer):
    cfg = getattr(tokenizer, "config", None)
    size = None
    if cfg is not None:
        try:
            size = int(cfg.model.vq_model.image_size)
        except Exception:
            size = None
    return size if size is not None and size > 0 else 256


def _images_to_tokens(tokenizer, images, expected_seq_len):
    image_size = _tokenizer_image_size(tokenizer)
    tokenizer_device = next(tokenizer.parameters()).device
    proc = []
    for img in images:
        crop = _center_crop_arr(img.convert("RGB"), image_size)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)
        proc.append(ten)
    x = torch.stack(proc, dim=0).to(tokenizer_device, non_blocking=True)

    with torch.no_grad():
        _, result_dict = tokenizer.encode(x)
    token_grid = result_dict["min_encoding_indices"].long()
    tokens = token_grid.view(token_grid.shape[0], -1)

    if tokens.shape[1] != int(expected_seq_len):
        raise ValueError(
            f"Token length mismatch: got {tokens.shape[1]}, expected {int(expected_seq_len)}. "
            "Check tokenizer/image size compatibility with the generator."
        )
    return tokens


def _iter_cc12m_tsv_batches(tsv_path, batch_size, rank=0, world_size=1, cache_dir=None, cache_images=True):
    batch_images = []
    batch_captions = []
    rank = int(rank)
    world_size = max(1, int(world_size))
    cache_dir = _ensure_cc12m_cache_dir(cache_dir) if bool(cache_images) else None

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if world_size > 1 and (line_idx % world_size) != rank:
                continue
            line = line.strip()
            if not line or "\t" not in line:
                continue
            image_url, caption = line.split("\t", 1)
            image_url = image_url.strip()
            caption = caption.strip()
            if not image_url or not caption:
                continue
            try:
                image = _fetch_cc12m_image(image_url, cache_dir=cache_dir, use_cache=bool(cache_images))
            except Exception:
                continue

            batch_images.append(image)
            batch_captions.append(caption)
            if len(batch_images) >= int(batch_size):
                out = (batch_images, batch_captions)
                batch_images, batch_captions = [], []
                yield out

    if len(batch_images) > 0:
        yield batch_images, batch_captions


def _iter_hf_stream_batches(
    urls,
    batch_size,
    rank=0,
    world_size=1,
    shard_retries=3,
    retry_sleep=2.0,
    cc12m_cache_dir=None,
    cc12m_cache_images=True,
):
    source = _normalize_dataset_source(urls)
    if source["kind"] == "cc12m_tsv":
        yield from _iter_cc12m_tsv_batches(
            tsv_path=source["value"],
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            cache_dir=cc12m_cache_dir,
            cache_images=cc12m_cache_images,
        )
        return

    batch_images = []
    batch_captions = []

    def _passthrough_nodesplitter(src, group=None):
        yield from src

    if urls is None or len(urls) == 0:
        urls = _default_hf_shard_urls(num_shards=69)

    if wds is None:
        raise RuntimeError("webdataset is required for DDP training, but it is not installed.")

    # In DDP, shard-level partitioning avoids every rank downloading all remote shards.
    urls = [str(u).strip() for u in urls if str(u).strip()]
    if int(world_size) > 1:
        urls = [u for i, u in enumerate(urls) if (i % int(world_size)) == int(rank)]
    if len(urls) == 0:
        return

    for shard_idx, url in enumerate(urls):
        shard = str(url).strip()
        if not shard:
            continue

        shard_ok = False
        last_error = None
        for attempt in range(1, int(shard_retries) + 1):
            try:
                dataset = wds.WebDataset(
                    [shard],
                    shardshuffle=False,
                    nodesplitter=_passthrough_nodesplitter,
                    workersplitter=wds.split_by_worker,
                ).decode("pil")
                for sample in dataset:
                    image = _extract_pil_image_from_sample(sample)
                    caption = _extract_caption_from_sample(sample)
                    if image is None or caption is None:
                        continue

                    batch_images.append(image)
                    batch_captions.append(caption)
                    if len(batch_images) >= int(batch_size):
                        out = (batch_images, batch_captions)
                        batch_images, batch_captions = [], []
                        yield out

                shard_ok = True
                break
            except Exception as exc:
                last_error = exc
                if attempt < int(shard_retries):
                    time.sleep(float(retry_sleep))

        if not shard_ok:
            continue

    if len(batch_images) > 0:
        yield batch_images, batch_captions


class TokenRegretCritic(nn.Module):
    def __init__(self, hidden_dim, text_dim, timestep_dim=32, mlp_dim=512, logits_topk=8, use_hidden=True):
        super().__init__()
        self.use_hidden = bool(use_hidden)
        self.logits_topk = int(logits_topk)
        self.timestep_dim = int(timestep_dim)
        in_dim = self.timestep_dim + self.logits_topk + 2 + text_dim
        if self.use_hidden:
            in_dim += hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

    def _timestep_embedding(self, timesteps):
        half = self.timestep_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.timestep_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _logit_features(self, logits):
        k = min(self.logits_topk, logits.shape[-1])
        topk = torch.topk(logits, k=k, dim=-1).values
        if k < self.logits_topk:
            topk = F.pad(topk, (0, self.logits_topk - k))
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1, keepdim=True)
        max_prob = probs.max(dim=-1, keepdim=True).values
        return torch.cat([topk, entropy, max_prob], dim=-1)

    def forward(self, hidden_states, logits, timesteps, text_features):
        bsz, seq_len, _ = logits.shape
        t_feat = self._timestep_embedding(timesteps).unsqueeze(1).expand(bsz, seq_len, -1)
        logit_feat = self._logit_features(logits)
        text_feat = text_features.unsqueeze(1).expand(bsz, seq_len, -1)
        chunks = [t_feat, logit_feat, text_feat]
        if self.use_hidden:
            chunks.insert(1, hidden_states)
        x = torch.cat(chunks, dim=-1)
        return self.net(x).squeeze(-1)


def build_token_regret_critic(model, use_hidden=True):
    hidden_dim = int(model.pos_embed.shape[-1])
    text_dim = int(model.pos_embed.shape[-1])
    return TokenRegretCritic(hidden_dim=hidden_dim, text_dim=text_dim, use_hidden=use_hidden)


@torch.no_grad()
def forward_features_vq(model, input_ids, condition, condition_pooled, aesthetic_score=None):
    embeddings = model.embeddings(input_ids)
    cond = model.text_embed_proj(condition)
    pooled = condition_pooled
    if model.micro_condition:
        pooled = model.concat_micro_cond(pooled, aesthetic_score)
    pooled = model.cond_pooled_proj(pooled)

    x = embeddings + model.pos_embed[:, :embeddings.shape[1]]
    for blk in model.blocks:
        cond, x = blk(x, cond, pooled.squeeze(1))
    x = model.norm(x, pooled.squeeze(1))
    logits = model.lm_head(x)
    return logits, x, pooled.squeeze(1)


def sample_masked_state(model, gt_tokens, timesteps):
    bsz, seq_len = gt_tokens.shape
    mask_ratio = get_masking_ratio(timesteps, model.mask_schedule_strategy)
    mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.0)
    num_masked = (seq_len * mask_ratio).round().clamp(min=1)
    randperm = torch.rand(bsz, seq_len, device=gt_tokens.device).argsort(dim=-1)
    masks = randperm < num_masked.unsqueeze(1)
    z_t = torch.where(masks, torch.full_like(gt_tokens, int(model.mask_token_id)), gt_tokens)
    return z_t, masks


def extract_token_nll(logits, target_tokens):
    return F.cross_entropy(logits.transpose(1, 2), target_tokens, reduction="none")


def select_token_subset(candidate_mask, sample_ratio, min_tokens=1):
    bsz, seq_len = candidate_mask.shape
    k = max(int(min_tokens), int(seq_len * float(sample_ratio)))
    k = min(seq_len, k)
    scores = torch.rand(bsz, seq_len, device=candidate_mask.device)
    scores = scores.masked_fill(~candidate_mask, -1.0)
    idx = scores.topk(k=k, dim=-1).indices
    valid = candidate_mask.gather(dim=-1, index=idx)
    return idx, valid


def pairwise_rank_loss(scores, regrets, valid_mask, margin=0.05):
    total = scores.new_tensor(0.0)
    count = 0
    for b in range(scores.shape[0]):
        keep = valid_mask[b]
        if keep.sum() < 2:
            continue
        s = scores[b][keep]
        r = regrets[b][keep]
        higher = (r.unsqueeze(1) - r.unsqueeze(0)) > 0
        if higher.any():
            loss_mat = F.relu(float(margin) - (s.unsqueeze(1) - s.unsqueeze(0)))
            total = total + loss_mat[higher].mean()
            count += 1
    if count == 0:
        return scores.new_tensor(0.0)
    return total / count


def save_critic_checkpoint(path, critic, optimizer, step, config_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config_dict,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser("Token Regret Critic DDP Trainer")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--per-gpu-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--token-sample-ratio", type=float, default=0.2)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--counterfactual-chunk-size", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/token_regret_critic")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--train-data-source", type=str, nargs="*", default=None)
    parser.add_argument("--cc12m-cache-dir", type=str, default=_default_cc12m_cache_dir())
    parser.add_argument("--disable-cc12m-cache", action="store_true")
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        local_rank = max(0, min(local_rank, gpu_count - 1))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    hf_cache_dir = os.path.abspath(os.environ.get("HF_CACHE_DIR", "hf_cache"))
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["OPENCLIP_CACHE_DIR"] = hf_cache_dir

    set_global_seed(args.seed + rank)

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

    class FeatureModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, condition, condition_pooled, aesthetic_score=None):
            return forward_features_vq(self.base_model, input_ids, condition, condition_pooled, aesthetic_score)

    feature_model = FeatureModel(model).to(device)
    critic = build_token_regret_critic(model, use_hidden=True).to(device)

    use_ddp = world_size > 1 and device.type == "cuda"
    if use_ddp:
        critic = DDP(critic, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(critic.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    start_step = 0
    if str(args.resume_checkpoint).strip():
        resume_path = str(args.resume_checkpoint).strip()
        ckpt = torch.load(resume_path, map_location=device)
        critic_for_load = critic.module if isinstance(critic, DDP) else critic
        critic_for_load.load_state_dict(ckpt["critic"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if is_main_process():
            print(f"Resumed from checkpoint: {resume_path} (step={start_step})")

    if args.train_data_source is None or len(args.train_data_source) == 0:
        urls = _default_hf_shard_urls()
    else:
        urls = [str(x).strip() for x in args.train_data_source if str(x).strip()]

    source = _normalize_dataset_source(urls)
    use_cc12m_cache = not bool(args.disable_cc12m_cache)
    cc12m_cache_dir = _ensure_cc12m_cache_dir(args.cc12m_cache_dir) if use_cc12m_cache else None
    if is_main_process() and source["kind"] == "cc12m_tsv":
        print(f"CC12M image cache: {'disabled' if not use_cc12m_cache else cc12m_cache_dir}")

    dataset_examples = _infer_total_examples_from_urls(urls)
    steps_per_epoch = None
    if dataset_examples is not None:
        steps_per_epoch = max(1, math.ceil(int(dataset_examples) / (int(args.per_gpu_batch_size) * max(world_size, 1))))

    total_steps = None if steps_per_epoch is None else int(steps_per_epoch) * int(args.num_epochs)
    if is_main_process():
        if total_steps is not None:
            print(f"Total iterations: {total_steps}")
        else:
            print("Total iterations: unknown")

    pbar = tqdm(total=total_steps, initial=start_step, desc="iterations", unit="iter", disable=not is_main_process())
    step = int(start_step)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {
        "num_epochs": int(args.num_epochs),
        "per_gpu_batch_size": int(args.per_gpu_batch_size),
        "world_size": int(world_size),
        "ddp": bool(use_ddp),
        "train_data_source": urls,
        "cc12m_cache_dir": cc12m_cache_dir,
        "cc12m_cache_enabled": bool(use_cc12m_cache),
        "resume_checkpoint": str(args.resume_checkpoint).strip() if str(args.resume_checkpoint).strip() else None,
    }
    if is_main_process():
        with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    default_aes = float(getattr(model, "sample_aesthetic_score", 6.5))

    for _ in range(int(args.num_epochs)):
        stream = _iter_hf_stream_batches(
            urls=urls,
            batch_size=int(args.per_gpu_batch_size),
            rank=rank,
            world_size=world_size,
            cc12m_cache_dir=cc12m_cache_dir,
            cc12m_cache_images=use_cc12m_cache,
        )
        for images, captions in stream:
            gt_tokens = _images_to_tokens(tokenizer, images, int(model.image_seq_len)).to(device, non_blocking=True)

            aes_scores = None
            if model.micro_condition:
                aes_scores = torch.full((gt_tokens.shape[0],), default_aes, device=device)

            with torch.no_grad():
                condition, condition_pooled = model.preprocess_condition(captions, clip_tokenizer, clip_encoder)
                condition = condition.to(device, non_blocking=True)
                condition_pooled = condition_pooled.to(device, non_blocking=True)
                timesteps = torch.rand((gt_tokens.shape[0],), device=device)
                z_t, _ = sample_masked_state(model, gt_tokens, timesteps)
                logits_orig, hidden_orig, text_feat = feature_model(z_t, condition, condition_pooled, aes_scores)

            candidate_mask = z_t.ne(int(model.mask_token_id))
            selected_idx, selected_valid = select_token_subset(candidate_mask, sample_ratio=float(args.token_sample_ratio), min_tokens=1)

            with torch.no_grad():
                nll_orig = extract_token_nll(logits_orig, gt_tokens)
                nll_orig_selected = nll_orig.gather(dim=-1, index=selected_idx)
                regrets = torch.zeros_like(nll_orig_selected)
                pair = torch.nonzero(selected_valid, as_tuple=False)
                for start in range(0, pair.shape[0], int(args.counterfactual_chunk_size)):
                    part = pair[start:start + int(args.counterfactual_chunk_size)]
                    if part.numel() == 0:
                        continue
                    b_idx = part[:, 0]
                    k_idx = part[:, 1]
                    tok_idx = selected_idx[b_idx, k_idx]

                    z_cf = z_t[b_idx].clone()
                    z_cf[torch.arange(z_cf.shape[0], device=z_cf.device), tok_idx] = int(model.mask_token_id)
                    cond_cf = condition[b_idx]
                    pooled_cf = condition_pooled[b_idx]
                    aes_cf = aes_scores[b_idx] if aes_scores is not None else None

                    logits_cf, _, _ = feature_model(z_cf, cond_cf, pooled_cf, aes_cf)
                    gt_cf = gt_tokens[b_idx]
                    nll_cf_all = extract_token_nll(logits_cf, gt_cf)
                    nll_cf_target = nll_cf_all[torch.arange(nll_cf_all.shape[0], device=z_cf.device), tok_idx]
                    regrets[b_idx, k_idx] = nll_orig_selected[b_idx, k_idx] - nll_cf_target
                targets = torch.tanh(regrets)

            pred = critic(
                hidden_states=hidden_orig,
                logits=logits_orig,
                timesteps=timesteps,
                text_features=text_feat,
            )
            pred_sel = pred.gather(dim=-1, index=selected_idx)

            denom = selected_valid.float().sum().clamp(min=1.0)
            loss_reg = (((pred_sel - targets) ** 2) * selected_valid.float()).sum() / denom
            loss_rank = pairwise_rank_loss(pred_sel, regrets, selected_valid, margin=float(args.rank_margin))
            loss = loss_reg + float(args.lambda_rank) * loss_rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)
            if is_main_process() and step % 10 == 0:
                pbar.set_postfix({"loss": float(loss.item()), "reg": float(loss_reg.item()), "rank": float(loss_rank.item())})

            if is_main_process() and step % int(args.save_every) == 0:
                critic_for_save = critic.module if isinstance(critic, DDP) else critic
                save_critic_checkpoint(os.path.join(args.output_dir, f"critic_step_{step}.pt"), critic_for_save, optimizer, step, cfg)

    if use_ddp:
        dist.barrier()

    critic_for_save = critic.module if isinstance(critic, DDP) else critic
    if is_main_process():
        save_critic_checkpoint(os.path.join(args.output_dir, "critic_last.pt"), critic_for_save, optimizer, step, cfg)
        print("Training done. Last checkpoint:", os.path.join(args.output_dir, "critic_last.pt"))

    pbar.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
