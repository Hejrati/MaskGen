import argparse
import json
import math
import os
import sys


def _early_cli_option(name, default):
    prefix = f"{name}="
    for idx, arg in enumerate(sys.argv):
        if arg == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


# Make the tokenizer attention backend explicit before importing modules that
# select their attention implementation at import time.
os.environ.setdefault(
    "MASKGEN_ATTENTION_MODE",
    _early_cli_option("--attention-backend", os.environ.get("GENEVAL_ATTENTION_BACKEND", "math")),
)

from PIL import Image
from tqdm import tqdm

import torch
import open_clip

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_VQ
from token_regrate.train_token_regret_ddp import (
    generate_image_vq_batch,
    get_config,
    load_trained_critic,
)
from token_regrate.utils import set_global_seed


def _str_to_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _none_or_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(value)


def _configure_paths(hf_cache_dir):
    hf_cache_dir = os.path.abspath(hf_cache_dir)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
    os.environ["OPENCLIP_CACHE_DIR"] = hf_cache_dir
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return hf_cache_dir


def _configure_torch_backend(attention_backend):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        backend = str(attention_backend).strip().lower()
        torch.backends.cuda.enable_flash_sdp(backend == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(backend == "xformers")
        torch.backends.cuda.enable_math_sdp(backend == "math")


def load_pretrained_stack(root_dir, clip_force_quick_gelu):
    tok_dir = os.path.join(root_dir, "turkeyju/tokenizer_tatitok_bl128_vq")
    gen_dir = os.path.join(root_dir, "turkeyju/generator_maskgen_vq_xl")

    print("Loading TA-TiTok tokenizer...")
    tatitok_vq_tokenizer = TATiTok.from_pretrained(
        pretrained_model_name_or_path=tok_dir,
        cache_dir=tok_dir,
    )
    tatitok_vq_tokenizer.eval()
    tatitok_vq_tokenizer.requires_grad_(False)

    print("Loading MaskGen-VQ generator...")
    maskgen_vq_generator = MaskGen_VQ.from_pretrained(
        pretrained_model_name_or_path=gen_dir,
        cache_dir=gen_dir,
    )
    maskgen_vq_generator.eval()
    maskgen_vq_generator.requires_grad_(False)

    print("Loading CLIP text encoder...")
    clip_encoder, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336",
        pretrained="openai",
        force_quick_gelu=bool(clip_force_quick_gelu),
    )
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval()
    clip_encoder.requires_grad_(False)

    return tatitok_vq_tokenizer, maskgen_vq_generator, clip_tokenizer, clip_encoder


def _load_geneval_metadata(metadata_jsonl_path):
    rows = []
    with open(metadata_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No prompts found in: " + str(metadata_jsonl_path))
    return rows


def _prepare_generation_config(
    batch_size,
    max_prompts,
    seed,
    guidance_scale,
    randomize_temperature,
    aesthetic_score,
    num_sample_steps,
    use_regret_remask,
    remask_ratio,
    refine_start_step,
    refine_loops,
    critic_use_hidden,
    repair_greedy,
    attention_backend,
    clip_force_quick_gelu,
    outdir=None,
):
    config = {
        "batch_size": int(batch_size),
        "max_prompts": _none_or_int(max_prompts),
        "seed": int(seed),
        "seed_policy": "fixed_per_prompt",
        "guidance_scale": float(guidance_scale),
        "randomize_temperature": float(randomize_temperature),
        "aesthetic_score": float(aesthetic_score),
        "num_sample_steps": int(num_sample_steps),
        "use_regret_remask": bool(use_regret_remask),
        "remask_ratio": float(remask_ratio),
        "refine_start_step": int(refine_start_step),
        "refine_loops": _none_or_int(refine_loops),
        "critic_use_hidden": bool(critic_use_hidden),
        "repair_greedy": bool(repair_greedy),
        "attention_backend": str(attention_backend),
        "clip_force_quick_gelu": bool(clip_force_quick_gelu),
        "generation_engine": "token_regret_wrapper_mimics_maskgen_generate",
    }
    if outdir is not None:
        config_path = os.path.join(outdir, "generation_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    return config


def _save_geneval_prompt_samples(prompt_idx, metadata, images, out_root, save_grid=True):
    prompt_dir = os.path.join(out_root, f"{int(prompt_idx):05d}")
    samples_dir = os.path.join(prompt_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    metadata_path = os.path.join(prompt_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True)

    for i, img in enumerate(images):
        img.save(os.path.join(samples_dir, f"{int(i):04d}.png"))

    if save_grid and images:
        n = len(images)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            grid.paste(img, (c * w, r * h))
        grid.save(os.path.join(prompt_dir, "grid.png"))


def generate_geneval_images_maskgen(
    model,
    tokenizer,
    clip_tokenizer,
    clip_encoder,
    metadata_jsonl,
    outdir,
    batch_size=4,
    max_prompts=None,
    seed=42,
    guidance_scale=12.0,
    randomize_temperature=1.5,
    aesthetic_score=6.5,
    num_sample_steps=16,
    use_regret_remask=False,
    critic=None,
    remask_ratio=0.10,
    refine_start_step=10,
    refine_loops=None,
    critic_use_hidden=True,
    repair_greedy=False,
    attention_backend="math",
    clip_force_quick_gelu=True,
):
    rows = _load_geneval_metadata(metadata_jsonl)
    if max_prompts is not None:
        rows = rows[: max(1, int(max_prompts))]

    os.makedirs(outdir, exist_ok=True)
    set_global_seed(int(seed))

    model_device = next(model.parameters()).device
    if next(clip_encoder.parameters()).device != model_device:
        clip_encoder = clip_encoder.to(model_device)
    if next(tokenizer.parameters()).device != model_device:
        tokenizer = tokenizer.to(model_device)

    if use_regret_remask:
        if critic is None:
            raise ValueError("use_regret_remask=True requires critic")
        if next(critic.parameters()).device != model_device:
            critic = critic.to(model_device)
        critic.eval()

    _prepare_generation_config(
        outdir=outdir,
        batch_size=batch_size,
        max_prompts=max_prompts,
        seed=seed,
        guidance_scale=guidance_scale,
        randomize_temperature=randomize_temperature,
        aesthetic_score=aesthetic_score,
        num_sample_steps=num_sample_steps,
        use_regret_remask=use_regret_remask,
        remask_ratio=remask_ratio,
        refine_start_step=refine_start_step,
        refine_loops=refine_loops,
        critic_use_hidden=critic_use_hidden,
        repair_greedy=repair_greedy,
        attention_backend=attention_backend,
        clip_force_quick_gelu=clip_force_quick_gelu,
    )

    for idx, meta in enumerate(tqdm(rows, desc="GenEval generation")):
        prompt = str(meta.get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError(f"Prompt is empty for metadata row {idx}")

        prompt_seed = int(seed)
        set_global_seed(int(prompt_seed))
        images = generate_image_vq_batch(
            prompts=[prompt] * int(batch_size),
            model=model,
            tokenizer=tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_encoder=clip_encoder,
            guidance_scale=float(guidance_scale),
            randomize_temperature=float(randomize_temperature),
            aesthetic_score=float(aesthetic_score),
            num_sample_steps=int(num_sample_steps),
            use_regret_remask=bool(use_regret_remask),
            critic=critic,
            remask_ratio=float(remask_ratio),
            refine_start_step=int(refine_start_step),
            refine_loops=refine_loops,
            critic_use_hidden=bool(critic_use_hidden),
            repair_greedy=bool(repair_greedy),
        )
        if len(images) != int(batch_size):
            raise RuntimeError(
                f"Expected {int(batch_size)} images for prompt {idx:05d}, got {len(images)}."
            )
        _save_geneval_prompt_samples(idx, meta, images, outdir)

    return outdir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "head"], required=True)
    parser.add_argument("--geneval-root", default="geneval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_CACHE_DIR", "hf_cache"))
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompts", type=_none_or_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=12.0)
    parser.add_argument("--randomize-temperature", type=float, default=2.0)
    parser.add_argument("--aesthetic-score", type=float, default=6.5)
    parser.add_argument("--num-sample-steps", type=int, default=16)
    parser.add_argument("--remask-ratio", type=float, default=0.10)
    parser.add_argument("--refine-start-step", type=int, default=10)
    parser.add_argument("--refine-loops", type=_none_or_int, default=None)
    parser.add_argument("--critic-use-hidden", action="store_true")
    parser.add_argument("--repair-greedy", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["math", "flash", "xformers"],
        default=os.environ.get("MASKGEN_ATTENTION_MODE", "math"),
    )
    parser.add_argument(
        "--clip-force-quick-gelu",
        type=_str_to_bool,
        default=_str_to_bool(os.environ.get("MASKGEN_CLIP_FORCE_QUICK_GELU", "true")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU GenEval generation.")

    hf_cache_dir = _configure_paths(args.hf_cache_dir)
    _configure_torch_backend(args.attention_backend)
    metadata_jsonl = os.path.join(
        os.path.abspath(args.geneval_root),
        "prompts",
        "evaluation_metadata.jsonl",
    )
    if not os.path.isfile(metadata_jsonl):
        raise FileNotFoundError(metadata_jsonl)

    print(f"Worker mode: {args.mode}")
    print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"OpenCLIP force_quick_gelu: {bool(args.clip_force_quick_gelu)}")

    tokenizer, model, clip_tokenizer, clip_encoder = load_pretrained_stack(
        hf_cache_dir,
        clip_force_quick_gelu=bool(args.clip_force_quick_gelu),
    )
    device = torch.device("cuda")
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    clip_encoder = clip_encoder.to(device)

    critic = None
    if args.mode == "head":
        config = get_config()
        resume_checkpoint = args.resume_checkpoint or config.runtime.resume_checkpoint
        critic = load_trained_critic(resume_checkpoint, model, use_hidden=bool(args.critic_use_hidden))
        critic = critic.to(device)
        print(f"Loaded critic head from: {resume_checkpoint}")

    generate_geneval_images_maskgen(
        model=model,
        tokenizer=tokenizer,
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        metadata_jsonl=metadata_jsonl,
        outdir=args.output_dir,
        batch_size=int(args.batch_size),
        max_prompts=args.max_prompts,
        seed=int(args.seed),
        guidance_scale=float(args.guidance_scale),
        randomize_temperature=float(args.randomize_temperature),
        aesthetic_score=float(args.aesthetic_score),
        num_sample_steps=int(args.num_sample_steps),
        use_regret_remask=args.mode == "head",
        critic=critic,
        remask_ratio=float(args.remask_ratio),
        refine_start_step=int(args.refine_start_step),
        refine_loops=args.refine_loops,
        critic_use_hidden=bool(args.critic_use_hidden),
        repair_greedy=bool(args.repair_greedy),
        attention_backend=str(args.attention_backend),
        clip_force_quick_gelu=bool(args.clip_force_quick_gelu),
    )


if __name__ == "__main__":
    main()
