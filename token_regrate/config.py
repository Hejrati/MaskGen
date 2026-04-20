from ml_collections import config_dict
import os

# Unified output directory for this config module.
OUTPUT_DIR = "outputs/token_regret_critic_overfit4"


def get_config():
    """Build the default token-regret training config."""
    config = config_dict.ConfigDict()

    config.experiment = config_dict.ConfigDict()
    # Human-readable run label stored in configs/checkpoints; paths below do not derive from it.
    config.experiment.name = "token_regret_critic"
    # Root directory for critic checkpoints, fixed training subset manifests, and run artifacts.
    config.experiment.output_dir = OUTPUT_DIR
    # Conventional log directory under output_dir; --output-dir rewrites this path.
    config.experiment.logging_dir = os.path.join(OUTPUT_DIR, "logs")

    config.training = config_dict.ConfigDict()
    # Base random seed; each DDP rank uses seed + rank for deterministic-but-distinct streams.
    config.training.seed = 42
    # Number of passes over the configured stream or fixed subset, unless max_steps stops first.
    config.training.num_epochs = 50
    # Hard cap on optimizer steps after resume; 0 means no explicit step cap.
    config.training.max_steps = 0
    # If >0, collect this many examples once and replay only that fixed subset; 0 uses the full dataset stream.
    config.training.max_train_images = 0
    # Per-process batch size; total global batch is this value times world_size.
    config.training.per_gpu_batch_size = 64
    # AdamW learning rate for the trainable token-regret critic only; generator/tokenizer stay frozen.
    config.training.learning_rate = 2e-4
    # Number of candidate token counterfactuals evaluated per chunk to control memory use.
    config.training.counterfactual_chunk_size = 512
    # Number of plain MaskGen denoise steps rolled forward after each token-remask counterfactual.
    config.training.counterfactual_rollout_steps = -1
    # Utility used to score baseline-vs-counterfactual quality; supports CE, cond-vs-uncond, and matched-vs-mismatched prompt variants.
    config.training.counterfactual_utility = "local_window_contrast"
    # Extra neighboring token radius included around the tested token for local-window utilities; 0 = token only.
    config.training.counterfactual_window_radius = 6
    # Number of mismatched batch prompts used by contrastive utilities; <=0 uses every available local-batch negative.
    config.training.counterfactual_contrast_negatives = 2
    # Temperature for InfoNCE-style contrastive prompt utility.
    config.training.counterfactual_contrast_temperature = 1.0
    # Contrastive utility reduction: "nce" uses log p(correct prompt), "neg_logsumexp" excludes the positive from the denominator.
    config.training.counterfactual_contrast_mode = "nce"
    # Use argmax instead of Gumbel sampling inside counterfactual rollouts.
    config.training.counterfactual_repair_greedy = True
    # Number of top cond-minus-uncond logit-gap features exposed to the regret critic; 0 disables them.
    config.training.critic_prompt_gap_topk = 8
    # Transform raw counterfactual regret targets before MSE; supported: "none"/"tanh"/"zscore".
    config.training.regret_target_transform = "tanh"
    # Notebook/control flag for DDP-only workflows; the training entrypoint does not currently branch on it.
    config.training.ddp_only = False
    # torch.distributed backend used by torchrun; "gloo" works on CPU/single-GPU, "nccl" is typical multi-GPU.
    config.training.dist_backend = "gloo"
    # Classifier-free guidance scale used when building training states and counterfactual logits.
    config.training.train_guidance_scale = 12.0
    # Gumbel/random sampling temperature for training-state and counterfactual denoising; 0 makes it deterministic.
    config.training.train_randomize_temperature = 0.0
    # Aesthetic conditioning scalar passed to MaskGen when the generator has micro-conditioning.
    config.training.train_aesthetic_score = 6.5
    # Fraction of the current MaskGen schedule mask budget that the critic may re-mask at guided steps.
    config.training.train_remask_ratio = 0.2
    # First sample step index where critic-guided refinement is allowed.
    config.training.train_refine_start_step = 8
    # Use argmax instead of Gumbel sampling while simulating the training-time MaskGen trajectory.
    config.training.train_repair_greedy = True
    # Fixed critic decision step for every training example; negative would sample uniformly over refine steps.
    config.training.fixed_rollout_step = -1
    # Replay earlier guided steps with the frozen EMA target critic when building training states.
    config.training.use_target_critic_replay = False
    # Max gradient norm for critic parameters; <=0 disables gradient clipping.
    config.training.grad_clip_norm = 1.0
    # Weight on the auxiliary pairwise ranking loss; 0 trains only the MSE regret regression loss.
    config.training.lambda_rank = 0.05
    # Margin used by the pairwise ranking loss when lambda_rank > 0.
    config.training.rank_margin = 0.1
    # Minimum target-regret gap required for a token pair to contribute to ranking loss.
    config.training.rank_gap_threshold = 0.1
    # EMA decay for the frozen target critic; closer to 1 updates it more slowly.
    config.training.target_critic_ema_decay = 0.995
    # Number of critic-guided decision steps after train_refine_start_step.
    config.training.refine_loops = 5
    # Save critic_step_N.pt and critic_last.pt every this many optimizer steps on rank 0.
    config.training.save_every = 1
    # Log metrics every this many optimizer steps on rank 0.
    config.training.log_every = 1

    config.inference = config_dict.ConfigDict()
    # Generation seed for notebook/evaluation helpers; kept equal to training.seed by default.
    config.inference.seed = config.training.seed
    # Inference-time number of critic-guided decision steps; defaults to the training schedule.
    config.inference.refine_loops = config.training.refine_loops
    # Inference-time critic remask budget fraction; defaults to the training remask ratio.
    config.inference.remask_ratio = config.training.train_remask_ratio
    # Inference-time first critic-guided step; defaults to the training start step.
    config.inference.refine_start_step = config.training.train_refine_start_step
    # Minimum predicted regret required for critic remasking; 0 means only predicted-positive gains reopen.
    config.inference.remask_min_score = 0.0
    # Use argmax instead of stochastic sampling during inference/evaluation generation.
    config.inference.repair_greedy = True

    config.dataset = config_dict.ConfigDict()
    # Dataset source mode: "hf" uses WebDataset shards, while local manifest paths are handled from source.
    config.dataset.mode = "hf"
    # Training data source override; empty string uses the built-in HF shard list for mode="hf".
    config.dataset.source = ""
    # Directory for cached CC12M downloads; empty string falls back to dataset/cc12m_image_cache.
    config.dataset.cc12m_cache_dir = ""
    # Disable on-disk CC12M image caching when True.
    config.dataset.disable_cc12m_cache = False
    # Number of parallel image-download workers for CC12M TSV manifests.
    config.dataset.cc12m_loader_workers = 48
    # Max queued CC12M download futures; 0 means loader_workers * 4.
    config.dataset.cc12m_loader_max_pending = 0
    # Number of already-loaded batches to prefetch from streaming datasets; 0 chooses an automatic small buffer.
    config.dataset.stream_prefetch_batches = 0

    config.model = config_dict.ConfigDict()
    # Local/HF path for the frozen MaskGen-VQ generator used to create token states and logits.
    config.model.generator_path = "hf_cache/turkeyju/generator_maskgen_vq_xl"
    # Local/HF path for the frozen TA-TiTok tokenizer used to encode images and decode tokens.
    config.model.tokenizer_path = "hf_cache/turkeyju/tokenizer_tatitok_bl128_vq"
    # OpenCLIP text encoder architecture name used for prompt conditioning.
    config.model.clip_name = "ViT-L-14-336"
    # OpenCLIP pretrained weights tag for the text encoder.
    config.model.clip_pretrained = "openai"
    # Request QuickGELU-compatible OpenCLIP weights/model construction.
    config.model.clip_force_quick_gelu = True
    # Total MaskGen denoising steps in training rollouts and generation.
    config.model.sample_steps = 16

    config.logging = config_dict.ConfigDict()
    # Enable TensorBoard and JSONL metric writing on rank 0.
    config.logging.enabled = True
    # Intended TensorBoard directory; --output-dir rewrites it, while TensorboardLogger currently derives it from output_dir/logs/tb.
    config.logging.tensorboard_dir = os.path.join(OUTPUT_DIR, "logs", "tb")
    # Intended JSONL metrics path; --output-dir rewrites it, while TensorboardLogger currently writes output_dir/logs/metrics.jsonl.
    config.logging.metrics_path = os.path.join(OUTPUT_DIR, "logs", "metrics.jsonl")

    config.runtime = config_dict.ConfigDict()
    # Checkpoint to resume critic/optimizer/EMA target from; missing files only warn and start fresh.
    config.runtime.resume_checkpoint = os.path.join(OUTPUT_DIR, "critic_last.pt")
    # Filled by setup_training_runtime from torch.distributed world size.
    config.runtime.world_size = 1
    # Filled by setup_training_runtime; True only when world_size > 1 and CUDA is available.
    config.runtime.ddp = False
    # Filled by setup_training_runtime from LOCAL_RANK and used to select the CUDA device.
    config.runtime.local_rank = 0

    return config
