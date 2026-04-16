import json
import os
import random
import numpy as np
from PIL import Image
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from token_regrate.nn import TokenRegretCritic



def _to_serializable_scalar(value):
    """Convert common numeric types/tensors into JSON-serializable Python scalars."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return float(value.detach().float().mean().item())
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


class TensorboardLogger:
    """Write training signals to TensorBoard and a JSONL metrics log file."""

    def __init__(self, log_dir, run_name="train", enabled=True, flush_secs=30):
        self.enabled = bool(enabled)
        self.log_dir = os.path.join(log_dir, str(run_name))
        self.tb_dir = os.path.join(self.log_dir, "tb")
        self.metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
        self.config_path = os.path.join(self.log_dir, "config.json")
        self.writer = None
        self._metrics_fp = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        if SummaryWriter is not None:
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=int(flush_secs))
        self._metrics_fp = open(self.metrics_path, "a", encoding="utf-8")

    def log_config(self, config_dict):
        """Persist config to JSON and add it as TensorBoard text for quick inspection."""
        if not self.enabled:
            return
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        if self.writer is not None:
            config_text = json.dumps(config_dict, indent=2, sort_keys=True)
            self.writer.add_text("config/json", f"<pre>{config_text}</pre>", global_step=0)
            self.writer.flush()

    def log_text(self, tag, text, step=0):
        """Log a text note to TensorBoard."""
        if not self.enabled or self.writer is None:
            return
        self.writer.add_text(str(tag), str(text), global_step=int(step))

    def log_metrics(self, step, metrics, prefix=""):
        """Log scalar metrics to TensorBoard and append JSONL records."""
        if not self.enabled:
            return
        step = int(step)
        row = {"step": step, "metrics": {}}
        for key, value in metrics.items():
            scalar = _to_serializable_scalar(value)
            if not isinstance(scalar, (int, float)):
                continue
            tag = f"{prefix}{key}" if prefix else str(key)
            if self.writer is not None:
                self.writer.add_scalar(tag, float(scalar), step)
            row["metrics"][str(key)] = float(scalar)
        if row["metrics"]:
            self._metrics_fp.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._metrics_fp.flush()

    def close(self):
        """Flush and close open logger resources."""
        if self._metrics_fp is not None:
            self._metrics_fp.flush()
            self._metrics_fp.close()
            self._metrics_fp = None
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def set_global_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_dist():
    """Return True when torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Return current process rank, defaulting to 0 in non-DDP mode."""
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    """Return world size, defaulting to 1 in non-DDP mode."""
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    """Return True for rank-0 process."""
    return get_rank() == 0


def setup_distributed(backend="gloo"):
    """Initialize distributed process group when launched with torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or is_dist():
        return
    # Use torchrun-provided rendezvous variables to initialize DDP.
    dist.init_process_group(backend=str(backend), init_method="env://")


def setup_training_runtime(config, dist_backend="gloo"):
    """Initialize device and distributed state for a training entrypoint."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config.runtime.local_rank = int(local_rank)

    if torch.cuda.is_available():
        local_rank = max(0, min(local_rank, gpu_count - 1))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    setup_distributed(backend=dist_backend)
    rank = get_rank()
    world_size = get_world_size()

    config.runtime.world_size = int(world_size)
    config.runtime.ddp = bool(world_size > 1 and device.type == "cuda")
    return device, rank, world_size, local_rank


def setup_cache_environment(cache_dir=None):
    """Configure local cache environment variables and return the cache directory."""
    hf_cache_dir = os.path.abspath(str(cache_dir or os.environ.get("HF_CACHE_DIR", "hf_cache")))
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["OPENCLIP_CACHE_DIR"] = hf_cache_dir
    return hf_cache_dir


def cleanup_distributed():
    """Destroy the distributed process group if it exists."""
    if is_dist():
        dist.destroy_process_group()

def _upgrade_scalar_critic_state_dict(state_dict, target_state_dict):
    """Map legacy action-value critic checkpoints to the scalar-regret head."""
    migrated = False
    state_dict = dict(state_dict)
    weight_key = "net.4.weight"
    bias_key = "net.4.bias"
    if (
        weight_key in state_dict
        and weight_key in target_state_dict
        and state_dict[weight_key].shape != target_state_dict[weight_key].shape
    ):
        old_weight = state_dict[weight_key]
        new_weight = torch.zeros_like(target_state_dict[weight_key])
        old_rows = int(old_weight.shape[0])
        new_rows = int(new_weight.shape[0])
        copy_rows = min(old_rows, new_rows)
        if old_rows == 1 and new_rows >= 2:
            new_weight[1].copy_(old_weight[0])
        elif old_rows >= 2 and new_rows == 1:
            new_weight[0].copy_(old_weight[1] - old_weight[0])
        else:
            new_weight[:copy_rows].copy_(old_weight[:copy_rows])
        state_dict[weight_key] = new_weight
        migrated = True
    if (
        bias_key in state_dict
        and bias_key in target_state_dict
        and state_dict[bias_key].shape != target_state_dict[bias_key].shape
    ):
        old_bias = state_dict[bias_key]
        new_bias = torch.zeros_like(target_state_dict[bias_key])
        old_rows = int(old_bias.shape[0])
        new_rows = int(new_bias.shape[0])
        copy_rows = min(old_rows, new_rows)
        if old_rows == 1 and new_rows >= 2:
            new_bias[1].copy_(old_bias[0])
        elif old_rows >= 2 and new_rows == 1:
            new_bias[0].copy_(old_bias[1] - old_bias[0])
        else:
            new_bias[:copy_rows].copy_(old_bias[:copy_rows])
        state_dict[bias_key] = new_bias
        migrated = True
    return state_dict, migrated


def load_critic_checkpoint(path, critic, optimizer=None, map_location="cpu", target_critic=None):
    """Load critic, optional optimizer, and optional EMA target critic state from checkpoint file."""
    ckpt = torch.load(path, map_location=map_location)
    critic_state, migrated_scalar_head = _upgrade_scalar_critic_state_dict(ckpt["critic"], critic.state_dict())
    critic.load_state_dict(critic_state, strict=True)
    if optimizer is not None and "optimizer" in ckpt and not migrated_scalar_head:
        optimizer.load_state_dict(ckpt["optimizer"])
    target_restored = False
    target_migrated_scalar_head = False
    if target_critic is not None and "target_critic" in ckpt:
        target_state, target_migrated_scalar_head = _upgrade_scalar_critic_state_dict(
            ckpt["target_critic"],
            target_critic.state_dict(),
        )
        target_critic.load_state_dict(target_state, strict=True)
        target_critic.eval()
        target_critic.requires_grad_(False)
        target_restored = True
    ckpt["critic_scalar_head_migrated"] = bool(migrated_scalar_head)
    ckpt["target_critic_restored"] = bool(target_restored)
    ckpt["target_critic_scalar_head_migrated"] = bool(target_migrated_scalar_head)
    return ckpt


def build_token_regret_critic(model, use_hidden=True):
    """Construct a TokenRegretCritic using generator embedding dimensions."""
    hidden_dim = int(model.pos_embed.shape[-1])
    text_dim = int(model.pos_embed.shape[-1])
    return TokenRegretCritic(hidden_dim=hidden_dim, text_dim=text_dim, use_hidden=bool(use_hidden))


def load_trained_critic(ckpt_path, model, use_hidden=True):
    """Build and load a frozen critic module on the model device."""
    target_device = next(model.parameters()).device
    critic = build_token_regret_critic(model, use_hidden=use_hidden).to(target_device)
    _ = load_critic_checkpoint(ckpt_path, critic, optimizer=None, map_location=target_device)
    critic.eval()
    critic.requires_grad_(False)
    return critic

def save_critic_checkpoint(path, critic, optimizer, step, config_dict, target_critic=None) -> None:
    """Persist critic/optimizer/EMA target state and training metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config_dict,
        "critic_output": "scalar_counterfactual_regret",
    }
    if target_critic is not None:
        payload["target_critic"] = target_critic.state_dict()
    torch.save(payload, path)

def prefetch_batches(iterable, prefetch_batches=4):
    """Prefetch iterable items on a background thread to overlap I/O and compute."""
    from queue import Queue
    from threading import Thread

    prefetch_batches = max(1, int(prefetch_batches))
    queue = Queue(maxsize=prefetch_batches)
    sentinel = object()
    error_box = {}

    def _worker():
        """Producer thread that fills queue and forwards exceptions."""
        try:
            for item in iterable:
                queue.put(item)
        except Exception as exc:
            error_box["exc"] = exc
        finally:
            queue.put(sentinel)

    thread = Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item

    thread.join()
    if "exc" in error_box:
        raise error_box["exc"]
    
    
def _center_crop_arr(pil_image, image_size):
    """Resize then center-crop a PIL image to a square size."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def images_to_tokens(tokenizer, images):
    """Convert PIL images into discrete tokenizer token grids."""
    image_size = int(tokenizer.config.dataset.preprocessing.crop_size)
    tokenizer_seq_len = int(tokenizer.config.model.vq_model.num_latent_tokens)

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

    if tokens.shape[1] != tokenizer_seq_len:
        raise ValueError(f"Token length mismatch: got {tokens.shape[1]}, expected {tokenizer_seq_len}")

    return tokens
