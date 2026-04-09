import json
import os
import random
import numpy as np

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

def load_critic_checkpoint(path, critic, optimizer=None, map_location="cpu"):
    """Load critic (and optional optimizer) state from checkpoint file."""
    ckpt = torch.load(path, map_location=map_location)
    critic.load_state_dict(ckpt["critic"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
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

def save_critic_checkpoint(path, critic, optimizer, step, config_dict) -> None: 
    """Persist critic/optimizer state and training metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config_dict,
    }
    torch.save(payload, path)

