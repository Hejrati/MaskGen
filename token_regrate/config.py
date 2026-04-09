from ml_collections import config_dict


def get_config():
    """Build the default token-regret training config."""
    config = config_dict.ConfigDict()

    config.experiment = config_dict.ConfigDict()
    config.experiment.name = "token_regret_critic"
    config.experiment.output_dir = "outputs/token_regret_critic"
    config.experiment.logging_dir = "outputs/token_regret_critic/logs"

    config.training = config_dict.ConfigDict()
    config.training.seed = 42
    config.training.num_epochs = 1
    config.training.per_gpu_batch_size = 128
    config.training.learning_rate = 2e-4
    config.training.lambda_rank = 0.1
    config.training.rank_margin = 0.05 # margin threshold for pairwise ranking loss
    config.training.token_sample_ratio = 0.2 # check whther exist in train loop
    config.training.counterfactual_chunk_size = 64 # check whther exist in train loop
    config.training.ddp_only = False
    config.training.dist_backend = "gloo"
    config.training.rollout_prob = 0.7
    config.training.train_guidance_scale = 12.0
    config.training.train_randomize_temperature = 1.5
    config.training.train_aesthetic_score = 6.5
    config.training.train_remask_ratio = 0.10 # check whther exist in train loop
    config.training.train_refine_start_step = 10
    config.training.margin_threshold = 0.20 # check whther exist in train loop
    config.training.neighborhood_radius = 0 # check whther exist in train loop
    config.training.utility_sign = 1.0 # check whther exist in train loop
    config.training.target_transform = "zscore_tanh" # check whther exist in train loop
    config.training.target_scale = 1.0 # check whther exist in train loop
    config.training.target_zclip = 3.0 # check whther exist in train loop
    config.training.refine_loops = 1
    config.training.save_every = 5 
    config.training.log_every = 1

    config.inference = config_dict.ConfigDict()
    config.inference.seed = config.training.seed
    config.inference.refine_loops = config.training.refine_loops
    config.inference.remask_ratio = config.training.train_remask_ratio
    config.inference.refine_start_step = config.training.train_refine_start_step

    config.dataset = config_dict.ConfigDict()
    config.dataset.mode = "cc12m"
    config.dataset.source = "/home/behzad/MaskGen/dataset/cc12m_image_cache"
    config.dataset.cc12m_cache_dir = "dataset/cc12m_image_cache"
    config.dataset.disable_cc12m_cache = False
    config.dataset.cc12m_loader_workers = 48
    config.dataset.cc12m_loader_max_pending = 0
    config.dataset.stream_prefetch_batches = 0

    config.model = config_dict.ConfigDict()
    config.model.generator_path = "hf_cache/turkeyju/generator_maskgen_vq_xl"
    config.model.tokenizer_path = "hf_cache/turkeyju/tokenizer_tatitok_bl128_vq"
    config.model.clip_name = "ViT-L-14-336"
    config.model.sample_steps = 16

    config.logging = config_dict.ConfigDict()
    config.logging.enabled = True
    config.logging.tensorboard_dir = "outputs/token_regret_critic/logs/tb"
    config.logging.metrics_path = "outputs/token_regret_critic/logs/metrics.jsonl"

    config.runtime = config_dict.ConfigDict()
    config.runtime.resume_checkpoint = "outputs/token_regret_critic/critic_step_35.pt"
    config.runtime.world_size = 1
    config.runtime.ddp = False
    config.runtime.local_rank = 0

    return config
