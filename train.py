"""
Moshi finetuning training script.

Supports both distributed training backends:
- torchrun + FSDP: Standard PyTorch Elastic with FullyShardedDataParallel
- mpirun + DDP: MPI-based with DistributedDataParallel

Usage:
    # Using torchrun (FSDP or DDP)
    torchrun --nproc_per_node=4 train.py --config configs/finetune.yaml

    # Using mpirun (DDP recommended)
    mpirun -np 4 python train.py --config configs/finetune.yaml
"""

import dataclasses
import logging
import os
import pprint
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import fire
import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import AdamW

# =============================================================================
# CRITICAL: Disable torch.compile/dynamo for FSDP compatibility
# PyTorch 2.x's inductor can cause stride mismatches with FSDP + gradient checkpointing
# Error: "AssertionError: expected stride 1125==1152 at dim=0"
# =============================================================================
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Completely disable dynamo compilation to avoid FSDP issues
torch._dynamo.disable()
# Also set environment variable as backup
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from finetune.args import TrainArgs
from finetune.scheduler import get_scheduler, get_two_rate_optimizer, get_current_lr
from finetune.checkpointing import Checkpointer, CheckpointManager
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import get_interleaved_tokenizer
from finetune.distributed import (
    avg_aggregate,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_mpi,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate, EvalReturnData, EvalSpeakerConditioningInfo
from finetune.loss import (
    compute_loss_with_mask,
    compute_audio_loss_per_speaker,
    AudioLossResult,
)
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.advanced_monitor import AdvancedTrainingMonitor
from finetune.monitoring.sample_saver import SampleSaver
from finetune.monitoring.dialogue_sample_saver import DialogueSampleSaver
from finetune.monitoring.research_logger import ResearchLogger
from finetune.monitoring.pretty_logger import PrettyLogger, get_gpu_memory_info
from finetune.monitoring.utils import set_logger
from finetune.monitoring.enhanced_evaluation import EnhancedEvaluationOrchestrator
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_distributed_model, get_unwrapped_model
from finetune.backbone import LMModelWrapper
from moshi.models import loaders

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def pad_codes_for_model(codes: torch.Tensor, target_codebooks: int, zero_token_id: int) -> torch.Tensor:
    """
    Pad codes to match model's expected number of codebooks.

    Moshiko weights were trained with n_q=16 (17 codebooks), but mono finetuning
    data only has 9 codebooks (1 text + 8 audio). We pad the remaining codebooks
    with zero_token_id (-1) which the model treats as "no input".

    Args:
        codes: Input codes tensor [B, K, T]
        target_codebooks: Number of codebooks the model expects (e.g., 17)
        zero_token_id: Special token for no-input positions (-1)

    Returns:
        Padded codes tensor [B, target_codebooks, T]
    """
    B, K, T = codes.shape

    if K == target_codebooks:
        return codes  # Already correct size

    if K > target_codebooks:
        raise ValueError(
            f"Data has {K} codebooks but model expects only {target_codebooks}. "
            f"Cannot truncate codebooks - check model configuration."
        )

    # Pad to target codebooks with zero_token_id
    pad_amount = target_codebooks - K
    padded = torch.nn.functional.pad(
        codes,
        (0, 0, 0, pad_amount),  # Pad codebook dimension (dim 1) on the right
        mode="constant",
        value=zero_token_id,
    )
    return padded


def train(config: str):
    """Main training entry point."""
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize distributed training (supports both torchrun and mpirun)
    launcher_type = init_distributed()

    # Initialize PrettyLogger for enhanced visual output
    pretty = PrettyLogger(rank=get_rank(), width=90)

    # Print beautiful startup banner
    pretty.print_banner(version="2.1", subtitle="Korean Speech Model Finetuning")

    if launcher_type != "single":
        set_device()
    else:
        pretty.print_warning("Running in single GPU mode. Use torchrun/mpirun for multi-GPU.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    # Check if we're potentially resuming from existing checkpoint
    # This is a preliminary check before CheckpointManager is initialized
    potential_resume = (
        args.checkpoint.enabled
        and args.checkpoint.resume_if_exist
        and run_dir.exists()
        and (run_dir / "checkpoints").exists()
    )

    # Check run_dir in distributed mode (both torchrun and MPI)
    if is_distributed():
        if run_dir.exists() and not args.overwrite_run_dir and not potential_resume:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir`, remove {run_dir}, "
                f"or enable resume_if_exist in checkpoint configuration."
            )
        elif run_dir.exists() and args.overwrite_run_dir and not potential_resume:
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)
        elif potential_resume:
            main_logger_info(f"Run dir exists with checkpoints, resume mode detected: {run_dir}")

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    # Print training configuration with PrettyLogger
    training_config = {
        "train_data": str(args.data.train_data) if args.data.train_data else "N/A",
        "eval_data": str(args.data.eval_data) if args.data.eval_data else None,
        "duration_sec": args.duration_sec,
        "shuffle": args.data.shuffle,
        "batch_size": args.batch_size,
        "num_microbatches": args.num_microbatches,
        "max_steps": args.max_steps,
        "max_norm": args.max_norm,
        "lr": args.optim.lr,
        "depformer_lr": args.optim.depformer_lr if args.optim.depformer_lr else "Same as LR",
        "weight_decay": args.optim.weight_decay,
        "beta1": args.optim.beta1,
        "beta2": args.optim.beta2,
        "eps": args.optim.eps,
        "scheduler_type": args.scheduler.type if hasattr(args, "scheduler") else "onecycle",
        "warmup_steps": args.scheduler.warmup_steps if hasattr(args, "scheduler") else "N/A",
        "min_lr": args.scheduler.min_lr if hasattr(args, "scheduler") else "N/A",
        "first_codebook_weight": args.first_codebook_weight_multiplier,
        "text_padding_weight": args.text_padding_weight,
        "world_size": get_world_size(),
    }
    pretty.print_training_config(training_config)

    # Print hardware info
    device_name = "NVIDIA A100 80GB" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    pretty.print_hardware_info(get_world_size(), device_name)

    # Print checkpoint info
    pretty.print_checkpoint_info(
        str(run_dir),
        args.ckpt_freq if args.do_ckpt else 0,
        args.num_ckpt_keep,
    )

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")

    # Handle local paths when hf_repo_id is None
    if args.moshi_paths.hf_repo_id is None:
        # All paths should be local - create CheckpointInfo directly
        moshi_path = Path(args.moshi_paths.moshi_path)
        mimi_path = Path(args.moshi_paths.mimi_path)
        tokenizer_path = Path(args.moshi_paths.tokenizer_path)

        # Validate that local files exist
        if not moshi_path.exists():
            raise FileNotFoundError(f"Moshi weights not found: {moshi_path}")
        if not mimi_path.exists():
            raise FileNotFoundError(f"Mimi weights not found: {mimi_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # Load config if provided
        lm_config = None
        raw_config = None
        if args.moshi_paths.config_path:
            config_path = Path(args.moshi_paths.config_path)
            if config_path.exists():
                import json
                raw_config = json.loads(config_path.read_text())
                lm_config = dict(raw_config)

        main_logger_info(f"Using local paths: moshi={moshi_path}, mimi={mimi_path}")
        checkpoint_info = loaders.CheckpointInfo(
            moshi_weights=moshi_path,
            mimi_weights=mimi_path,
            tokenizer=tokenizer_path,
            lm_config=lm_config,
            raw_config=raw_config,
        )
    else:
        # Use HuggingFace repo
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo=args.moshi_paths.hf_repo_id,
            moshi_weights=args.moshi_paths.moshi_path,
            mimi_weights=args.moshi_paths.mimi_path,
            tokenizer=args.moshi_paths.tokenizer_path,
            config_path=args.moshi_paths.config_path,
        )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and wrap model (FSDP or DDP based on config), prepare interleaver for audio/text tokens.
    model = get_distributed_model(args, checkpoint_info)

    # Get unwrapped model for attribute access (needed for DDP which wraps model.module)
    unwrapped_model = get_unwrapped_model(model)

    # Print model configuration with PrettyLogger
    linears_count = len(unwrapped_model.linears) if hasattr(unwrapped_model, 'linears') else 0
    depformer_in_count = len(unwrapped_model.depformer_in) if hasattr(unwrapped_model, 'depformer_in') else 0
    depformer_norms_count = len(unwrapped_model.depformer_norms) if hasattr(unwrapped_model, 'depformer_norms') else 0

    model_config = {
        "model_type": "Moshiko" + (" (User Stream Extended)" if args.korean.enable_user_stream else ""),
        "n_q": unwrapped_model.n_q,
        "dep_q": unwrapped_model.dep_q,
        "linears": linears_count,
        "depformer_in": depformer_in_count,
        "depformer_norms": depformer_norms_count,
        "num_codebooks": unwrapped_model.num_codebooks,
        "audio_offset": unwrapped_model.audio_offset,
        "zero_token_id": unwrapped_model.zero_token_id,
        "full_finetuning": args.full_finetuning,
        "lora_enabled": args.lora.enable,
        "lora_rank": args.lora.rank if args.lora.enable else "N/A",
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    pretty.print_model_info(model_config)

    # CRITICAL: Architecture validation for User Stream mode
    if args.korean.enable_user_stream:
        expected_dep_q = 16
        expected_linears = 16
        expected_norms = 16
        if unwrapped_model.dep_q != expected_dep_q or linears_count != expected_linears or depformer_norms_count != expected_norms:
            pretty.print_error(
                "Architecture Mismatch for User Stream Mode!",
                f"Expected dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms} | "
                f"Actual dep_q={unwrapped_model.dep_q}, linears={linears_count}, depformer_norms={depformer_norms_count}"
            )
            raise RuntimeError(
                f"Model architecture mismatch for user stream mode!\n"
                f"  Config: enable_user_stream=True\n"
                f"  Expected: dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms}\n"
                f"  Actual:   dep_q={unwrapped_model.dep_q}, linears={linears_count}, depformer_norms={depformer_norms_count}\n\n"
                f"This error means the model was not properly extended for user stream training.\n"
                f"Please check wrapped_model.py logs for [CONFIG CHECK] and [ARCHITECTURE] messages."
            )
        else:
            pretty.print_success(f"User Stream architecture verified: dep_q={unwrapped_model.dep_q}, linears={linears_count}, depformer_norms={depformer_norms_count}")

    if unwrapped_model.n_q != 16 or unwrapped_model.num_codebooks != 17:
        pretty.print_warning(f"Expected n_q=16, num_codebooks=17. Data will be padded to {unwrapped_model.num_codebooks} codebooks.")

    # DIAGNOSTIC: Check model weights for NaN/Inf
    # NOTE: Skip detailed diagnostics for FSDP models - accessing sharded parameters
    # triggers expensive all-gather collective operations that can cause hangs.
    pretty.print_subsection("Weight Diagnostics", "🔍")

    # Check if model is FSDP wrapped (parameters are sharded)
    is_fsdp = isinstance(model, FullyShardedDataParallel)

    if is_fsdp:
        # For FSDP, only count parameters without accessing data
        # Parameter data is sharded, so NaN/Inf checks would require all-gather
        total_params = sum(1 for _ in model.parameters())
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        pretty.print_info(
            f"FSDP mode: {total_params:,} params ({trainable_params:,} trainable). "
            f"Skipping NaN/Inf check (sharded parameters)."
        )
    else:
        # For single GPU or DDP, perform full diagnostics
        nan_params = []
        inf_params = []
        zero_params = []
        total_params = 0
        trainable_params = 0
        for name, param in unwrapped_model.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
            if param.numel() > 0 and (param == 0).all():
                zero_params.append(name)

        if nan_params:
            pretty.print_error(f"NaN found in {len(nan_params)} parameters", str(nan_params[:3]))
        if inf_params:
            pretty.print_error(f"Inf found in {len(inf_params)} parameters", str(inf_params[:3]))
        if zero_params and len(zero_params) > 10:
            pretty.print_warning(f"Found {len(zero_params)} all-zero parameters")
        if not nan_params and not inf_params:
            pretty.print_success(f"All {total_params:,} parameters valid ({trainable_params:,} trainable)")

    spm = checkpoint_info.get_text_tokenizer()

    # Select tokenizer based on Korean finetuning configuration
    enable_user_stream = args.korean.enable_user_stream
    full_duplex_input = args.korean.full_duplex_input

    # Determine if stereo data is used (for padding logic)
    use_stereo_data = enable_user_stream or full_duplex_input

    # Log Korean config for debugging
    # Extract interleaver configuration from YAML (args.korean.interleaver)
    interleaver_cfg = args.korean.interleaver
    main_logger_info("=" * 60)
    main_logger_info("[KOREAN CONFIG]")
    main_logger_info(f"  enable_user_stream: {args.korean.enable_user_stream}")
    main_logger_info(f"  full_duplex_input: {args.korean.full_duplex_input}")
    main_logger_info(f"  initialized_model_path: {args.korean.initialized_model_path}")
    main_logger_info(f"  korean_tokenizer_path: {args.korean.korean_tokenizer_path}")
    main_logger_info(f"  korean_tokenizer_type: {args.korean.korean_tokenizer_type}")
    main_logger_info("[INTERLEAVER CONFIG]")
    main_logger_info(f"  keep_main_only: {interleaver_cfg.keep_main_only}")
    main_logger_info(f"  keep_and_shift: {interleaver_cfg.keep_and_shift}")
    main_logger_info(f"  adaptive_distribute: {interleaver_cfg.adaptive_distribute}")
    main_logger_info(f"  warn_on_overflow: {interleaver_cfg.warn_on_overflow}")
    main_logger_info(f"  character_level_interpolation: {interleaver_cfg.character_level_interpolation}")
    main_logger_info(f"  main_speaker_label: {interleaver_cfg.main_speaker_label}")

    # =========================================================================
    # Unified Segment Filtering Configuration (5-Layer System)
    # =========================================================================
    segment_filtering_cfg = args.korean.segment_filtering

    main_logger_info("=" * 60)
    main_logger_info("[UNIFIED SEGMENT FILTERING - 5 Layer System]")
    main_logger_info("=" * 60)

    # Layer 1: Case Control
    cc = segment_filtering_cfg.case_control
    main_logger_info(f"  Layer 1 (Case Control): {'ENABLED' if cc.enabled else 'DISABLED'}")
    if cc.enabled:
        main_logger_info(f"    allow_case1 (full dialogue): {cc.allow_case1}")
        main_logger_info(f"    allow_case2 (monologue): {cc.allow_case2}")
        main_logger_info(f"    allow_case3-5 (invalid): {cc.allow_case3}/{cc.allow_case4}/{cc.allow_case5}")

    # Layer 2: Quality
    q = segment_filtering_cfg.quality
    main_logger_info(f"  Layer 2 (Quality): {'ENABLED' if q.enabled else 'DISABLED'}")
    if q.enabled:
        main_logger_info(f"    min_moshi_words: {q.min_moshi_words}")
        main_logger_info(f"    min_moshi_duration_sec: {q.min_moshi_duration_sec}")
        main_logger_info(f"    min_user_words: {q.min_user_words}")

    # Layer 3: Preferences
    p = segment_filtering_cfg.preferences
    main_logger_info(f"  Layer 3 (Preferences): {'ENABLED' if p.enabled else 'DISABLED'}")

    # Layer 4: Role Swapping
    rs = segment_filtering_cfg.role_swapping
    main_logger_info(f"  Layer 4 (Role Swapping): {'ENABLED' if rs.enabled else 'DISABLED'}")
    if rs.enabled:
        main_logger_info(f"    yield_both: {rs.yield_both}")
        main_logger_info(f"    recheck_after_swap: {rs.recheck_after_swap}")

    # Layer 5: Logging
    lg = segment_filtering_cfg.logging
    main_logger_info(f"  Layer 5 (Logging): verbosity={lg.verbosity}, save_to_file={lg.save_to_file}")

    main_logger_info("=" * 60)

    # Training Mode Selection:
    # - USER-STREAM: enable_user_stream=True → 17 codebooks, dep_q=16 (predict user audio)
    # - FULL-DUPLEX: full_duplex_input=True → 17 codebooks, dep_q=8 (user audio as context only)
    # - MONOLOGUE: both False → 9 codebooks, dep_q=8 (mono training)
    if enable_user_stream:
        main_logger_info("=" * 60)
        main_logger_info("[USER-STREAM MODE]")
        main_logger_info("  Stereo input: 17 codebooks (1 text + 8 moshi + 8 user)")
        main_logger_info("  Output: dep_q=16 (predict both Moshi AND User audio)")
        main_logger_info("=" * 60)
    elif full_duplex_input:
        main_logger_info("=" * 60)
        main_logger_info("[FULL-DUPLEX MODE] (Original Moshi / J-Moshi default)")
        main_logger_info("  Stereo input: 17 codebooks (1 text + 8 moshi + 8 user)")
        main_logger_info("  Output: dep_q=8 (predict Moshi audio only, user as context)")
        main_logger_info("=" * 60)
    else:
        main_logger_info("=" * 60)
        main_logger_info("[MONOLOGUE MODE]")
        main_logger_info("  Mono input: 9 codebooks (1 text + 8 moshi)")
        main_logger_info("  Output: dep_q=8 (predict Moshi audio only)")
        main_logger_info("=" * 60)

    # =========================================================================
    # Stage Pretrained Model Loading (before speaker conditioning)
    # =========================================================================
    # This loads weights from a previous training stage while preserving
    # new parameters (e.g., speaker_conditioner) with their random init.
    # Must be done BEFORE speaker conditioning setup to ensure base model
    # weights are loaded correctly.
    if hasattr(args, "pretrained") and args.pretrained.enabled:
        from finetune.pretrained_loader import load_pretrained_weights, verify_pretrained_loading

        main_logger_info("=" * 60)
        main_logger_info("[STAGE PRETRAINED LOADING]")
        main_logger_info(f"  Source: {args.pretrained.path}")
        main_logger_info(f"  Checkpoint dir: {args.pretrained.checkpoint_dir}")
        main_logger_info(f"  Strict mode: {args.pretrained.strict}")
        main_logger_info(f"  Expected new modules: {args.pretrained.expected_new_modules}")
        main_logger_info("=" * 60)

        try:
            loaded_count, skipped_count, new_params = load_pretrained_weights(
                model=model,
                args=args.pretrained,
                run_dir=Path(args.run_dir).parent if args.run_dir else None,
            )

            main_logger_info(f"  Loaded parameters: {loaded_count:,}")
            main_logger_info(f"  Skipped (not in model): {skipped_count:,}")
            main_logger_info(f"  New parameters (initialized): {len(new_params):,}")

            # Verify loading was successful
            verify_pretrained_loading(unwrapped_model, loaded_count, new_params)

            # CRITICAL: Disable checkpoint resume when pretrained loading is used
            # We want to start from step 0 with fresh optimizer/scheduler
            if args.checkpoint.resume_if_exist:
                main_logger_info("[PRETRAINED] Disabling checkpoint resume - starting from step 0")
                args.checkpoint.resume_if_exist = False
            if args.checkpoint.resume_from:
                main_logger_info("[PRETRAINED] Clearing resume_from - starting from step 0")
                args.checkpoint.resume_from = None

            main_logger_info("=" * 60)

        except FileNotFoundError as e:
            pretty.print_error("Pretrained checkpoint not found", str(e))
            raise
        except RuntimeError as e:
            pretty.print_error("Pretrained loading failed (strict mode)", str(e))
            raise

    # =========================================================================
    # Phase 2: Speaker Conditioning Configuration
    # =========================================================================
    speaker_conditioning_config = None
    if hasattr(args, "speaker") and args.speaker is not None and args.speaker.enabled:
        main_logger_info("=" * 60)
        main_logger_info("[SPEAKER CONDITIONING ENABLED]")
        main_logger_info(f"  Method: {args.speaker.method}")
        main_logger_info(f"  Reference duration: [{args.speaker.reference_sampler.min_duration_sec}, "
                        f"{args.speaker.reference_sampler.max_duration_sec}]s")
        main_logger_info(f"  Encoder: {args.speaker.encoder.encoder_type}")
        main_logger_info(f"  Conditioner output_dim: {args.speaker.conditioner.output_dim}")
        main_logger_info("=" * 60)

        speaker_conditioning_config = {
            "enabled": True,
            "min_duration_sec": args.speaker.reference_sampler.min_duration_sec,
            "max_duration_sec": args.speaker.reference_sampler.max_duration_sec,
            "target_sample_rate": args.speaker.reference_sampler.target_sample_rate,
        }

    # Initialize speaker encoder and conditioner if enabled
    speaker_encoder = None
    speaker_conditioner = None
    if speaker_conditioning_config is not None and speaker_conditioning_config.get("enabled", False):
        try:
            from finetune.modules import (
                create_speaker_encoder,
                SpeakerEncoderConfig,
                SpeakerConditioner,
                SpeakerConditionerConfig,
            )

            # Create speaker encoder
            encoder_config = SpeakerEncoderConfig(
                encoder_type=args.speaker.encoder.encoder_type,
                pretrained_path=args.speaker.encoder.pretrained_path,
                freeze=args.speaker.encoder.freeze,
                output_dim=args.speaker.encoder.output_dim,
                sample_rate=args.speaker.encoder.sample_rate,
                normalize_embedding=args.speaker.encoder.normalize_embedding,
            )
            speaker_encoder = create_speaker_encoder(encoder_config)
            speaker_encoder = speaker_encoder.cuda()
            if args.speaker.encoder.freeze:
                speaker_encoder.freeze()
            main_logger_info(f"  Speaker encoder loaded: {args.speaker.encoder.encoder_type}")

            # Create speaker conditioner
            conditioner_config = SpeakerConditionerConfig(
                input_dim=args.speaker.encoder.output_dim,
                output_dim=args.speaker.conditioner.output_dim,
                initial_scale=args.speaker.conditioner.initial_scale,
                use_layernorm=args.speaker.conditioner.use_layernorm,
                dropout=args.speaker.conditioner.dropout,
                learnable_scale=args.speaker.conditioner.learnable_scale,
                scale_mode=args.speaker.conditioner.scale_mode,
            )
            speaker_conditioner = SpeakerConditioner(conditioner_config)
            speaker_conditioner = speaker_conditioner.cuda()
            main_logger_info(f"  Speaker conditioner loaded: {args.speaker.encoder.output_dim} -> {args.speaker.conditioner.output_dim}")

            # Set conditioner on model wrapper
            # When speaker conditioning is enabled, _should_use_modular_backbone() ensures
            # LMModelWrapper is used, which supports set_speaker_conditioner()
            #
            # Note: unwrapped_model may be FSDP-wrapped, so we need to access the inner module
            # FSDP wraps the model but we can still call methods through it or via .module
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Determine the actual model to configure
            if isinstance(unwrapped_model, FSDP):
                # FSDP wraps LMModelWrapper - access inner module for type check
                # PyTorch 2.x uses _fsdp_wrapped_module, older versions use module
                if hasattr(unwrapped_model, '_fsdp_wrapped_module'):
                    inner_module = unwrapped_model._fsdp_wrapped_module
                elif hasattr(unwrapped_model, 'module'):
                    inner_module = unwrapped_model.module
                else:
                    inner_module = unwrapped_model

                if not isinstance(inner_module, LMModelWrapper):
                    raise RuntimeError(
                        f"Speaker conditioning requires LMModelWrapper inside FSDP, but got "
                        f"{type(inner_module).__name__}."
                    )
                # Set conditioner on the inner LMModelWrapper
                inner_module.set_speaker_conditioner(speaker_conditioner)
                inner_module.enable_speaker_conditioning()
                main_logger_info("  Speaker conditioning integrated with FSDP-wrapped LMModelWrapper")
            elif isinstance(unwrapped_model, LMModelWrapper):
                unwrapped_model.set_speaker_conditioner(speaker_conditioner)
                unwrapped_model.enable_speaker_conditioning()
                main_logger_info("  Speaker conditioning integrated with LMModelWrapper")
            else:
                raise RuntimeError(
                    "Speaker conditioning requires LMModelWrapper, but got "
                    f"{type(unwrapped_model).__name__}. This should not happen - "
                    "_should_use_modular_backbone() should return True when speaker "
                    "conditioning is enabled."
                )

        except ImportError as e:
            main_logger_info(f"  WARNING: Speaker conditioning modules not available: {e}")
            main_logger_info("  Continuing without speaker conditioning")
            speaker_conditioning_config = None
            speaker_encoder = None
            speaker_conditioner = None

    # =========================================================================
    # Phase 2: Audio Prompt Modules for Training AND Evaluation
    # =========================================================================
    # Create TWO audio prompt modules:
    # 1. train_audio_prompt_module: Random sampling for training (avoid overlap)
    # 2. eval_audio_prompt_module: Deterministic sampling for evaluation
    #
    # CRITICAL: Zero-shot speaker adaptation requires prompting during TRAINING.
    # The reference audio+text is prepended to the sequence, and the model learns
    # to continue generating in the reference speaker's voice.
    # =========================================================================
    train_audio_prompt_module = None
    eval_audio_prompt_module = None

    if (hasattr(args, "speaker") and args.speaker is not None and
        args.speaker.enabled and
        hasattr(args.speaker, "audio_prompt") and
        args.speaker.audio_prompt.enable):
        try:
            from finetune.modules.audio_prompt import (
                create_audio_prompt_module,
                AudioPromptConfig,
            )

            # -----------------------------------------------------------------
            # TRAINING Audio Prompt Config (Random Sampling)
            # -----------------------------------------------------------------
            # - deterministic=False: Random segment selection
            # - avoid_overlap=False: Allow overlap for natural conversation flow
            #   (Reference and training can share frames - more natural learning)
            # - sample_strategy="random": Random position in sequence
            # -----------------------------------------------------------------
            train_prompt_config = AudioPromptConfig(
                enable=True,
                mode=args.speaker.audio_prompt.mode,
                min_duration_sec=args.speaker.audio_prompt.min_duration_sec,
                max_duration_sec=args.speaker.audio_prompt.max_duration_sec,
                sample_strategy="random",  # Random for training
                avoid_overlap=False,  # Allow overlap for natural conversation flow
                # Non-deterministic for training (random sampling)
                deterministic=False,
                fixed_duration_sec=getattr(args.speaker.audio_prompt, 'fixed_duration_sec', 10.0),
                use_word_count=getattr(args.speaker.audio_prompt, 'use_word_count', False),
                fixed_word_count=getattr(args.speaker.audio_prompt, 'fixed_word_count', 20),
            )

            train_audio_prompt_module = create_audio_prompt_module(
                config=train_prompt_config,
                frame_rate=mimi.frame_rate,  # 12.5 Hz
                audio_offset=unwrapped_model.audio_offset,
                text_padding_token_id=unwrapped_model.text_padding_token_id,
            )
            train_audio_prompt_module = train_audio_prompt_module.cuda()

            main_logger_info(f"  TRAIN Audio prompt module initialized:")
            main_logger_info(f"    Mode: {train_prompt_config.mode}")
            main_logger_info(f"    Deterministic: {train_prompt_config.deterministic}")
            main_logger_info(f"    Duration: {train_prompt_config.min_duration_sec}-{train_prompt_config.max_duration_sec}s")
            main_logger_info(f"    Strategy: {train_prompt_config.sample_strategy}")
            main_logger_info(f"    Avoid overlap: {train_prompt_config.avoid_overlap}")

            # -----------------------------------------------------------------
            # EVALUATION Audio Prompt Config (Deterministic Sampling)
            # -----------------------------------------------------------------
            # - deterministic=True: Fixed position (reproducible)
            # - avoid_overlap=False: No exclusion needed for eval
            # - sample_strategy="start": Always from beginning
            # -----------------------------------------------------------------
            eval_prompt_config = AudioPromptConfig(
                enable=True,
                mode=args.speaker.audio_prompt.mode,
                min_duration_sec=args.speaker.audio_prompt.min_duration_sec,
                max_duration_sec=args.speaker.audio_prompt.max_duration_sec,
                sample_strategy=getattr(args.speaker.audio_prompt, 'sample_strategy', 'start'),
                avoid_overlap=False,  # No exclusion needed for eval
                # Deterministic settings for evaluation
                deterministic=getattr(args.speaker.audio_prompt, 'deterministic', True),
                fixed_duration_sec=getattr(args.speaker.audio_prompt, 'fixed_duration_sec', 10.0),
                use_word_count=getattr(args.speaker.audio_prompt, 'use_word_count', False),
                fixed_word_count=getattr(args.speaker.audio_prompt, 'fixed_word_count', 20),
            )

            eval_audio_prompt_module = create_audio_prompt_module(
                config=eval_prompt_config,
                frame_rate=mimi.frame_rate,  # 12.5 Hz
                audio_offset=unwrapped_model.audio_offset,
                text_padding_token_id=unwrapped_model.text_padding_token_id,
            )
            eval_audio_prompt_module = eval_audio_prompt_module.cuda()

            main_logger_info(f"  EVAL Audio prompt module initialized:")
            main_logger_info(f"    Mode: {eval_prompt_config.mode}")
            main_logger_info(f"    Deterministic: {eval_prompt_config.deterministic}")
            main_logger_info(f"    Fixed duration: {eval_prompt_config.fixed_duration_sec}s")
            main_logger_info(f"    Strategy: {eval_prompt_config.sample_strategy}")

        except Exception as e:
            main_logger_info(f"  WARNING: Could not create audio prompt modules: {e}")
            import traceback
            main_logger_info(f"  Traceback: {traceback.format_exc()}")
            train_audio_prompt_module = None
            eval_audio_prompt_module = None

    # For backward compatibility, keep audio_prompt_module as alias for eval
    audio_prompt_module = eval_audio_prompt_module

    # Create tokenizer (stereo for USER-STREAM or FULL-DUPLEX, mono for MONOLOGUE)
    interleaved_tokenizer = get_interleaved_tokenizer(
        mimi=mimi,
        spm=spm,
        duration_sec=args.duration_sec,
        text_padding_token_id=unwrapped_model.text_padding_token_id,
        end_of_text_padding_id=unwrapped_model.end_of_text_padding_id,
        zero_token_id=unwrapped_model.zero_token_id,
        enable_user_stream=enable_user_stream,
        full_duplex_input=full_duplex_input,
        # Interleaver configuration
        keep_main_only=interleaver_cfg.keep_main_only,
        keep_and_shift=interleaver_cfg.keep_and_shift,
        adaptive_distribute=interleaver_cfg.adaptive_distribute,
        warn_on_overflow=interleaver_cfg.warn_on_overflow,
        character_level_interpolation=interleaver_cfg.character_level_interpolation,
        main_speaker_label=interleaver_cfg.main_speaker_label,
        # Unified segment filtering configuration (5-layer system)
        segment_filtering=segment_filtering_cfg if use_stereo_data else None,
        run_dir=args.run_dir if use_stereo_data else None,
        # Phase 2: Speaker conditioning
        speaker_conditioning_config=speaker_conditioning_config,
    )

    # 5. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )

    # =========================================================================
    # CRITICAL FIX: Use factory function to create fresh eval_data_loader
    # =========================================================================
    # Problem: build_data_loader returns a generator (uses yield), which can
    # only be iterated once. After the first evaluation, the generator is
    # exhausted and subsequent evaluations get no data.
    #
    # Solution: Create a factory function that returns a new data loader
    # each time evaluation is needed.
    # =========================================================================
    def create_eval_data_loader():
        """Factory function to create a fresh eval data loader for each evaluation."""
        return build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )

    # 5.5 Initialize Advanced Training Monitor
    advanced_monitor = None
    if hasattr(args, "monitoring") and args.monitoring is not None:
        main_logger_info("=" * 60)
        main_logger_info("[ADVANCED MONITORING ENABLED]")

        # Convert monitoring args to dict for the monitor
        monitoring_config = {
            "text_evaluation": {
                "enabled": args.monitoring.text_evaluation.enabled,
                "max_prediction_samples": args.monitoring.text_evaluation.max_prediction_samples,
                "normalize_text": args.monitoring.text_evaluation.normalize_text,
            },
            "codebook_analysis": {
                "enabled": args.monitoring.codebook_analysis.enabled,
                "log_entropy": args.monitoring.codebook_analysis.log_entropy,
            },
            "gradient_monitoring": {
                "enabled": args.monitoring.gradient_monitoring.enabled,
                "exploding_threshold": args.monitoring.gradient_monitoring.exploding_threshold,
                "vanishing_threshold": args.monitoring.gradient_monitoring.vanishing_threshold,
                "alert_on_nan": args.monitoring.gradient_monitoring.alert_on_nan,
                "alert_on_inf": args.monitoring.gradient_monitoring.alert_on_inf,
                "log_per_layer": args.monitoring.gradient_monitoring.log_per_layer,
            },
        }

        advanced_monitor = AdvancedTrainingMonitor(
            tokenizer=spm,
            text_padding_token_id=unwrapped_model.text_padding_token_id,
            end_of_text_padding_id=unwrapped_model.end_of_text_padding_id,
            num_codebooks=unwrapped_model.dep_q,  # Number of audio codebooks
            first_codebook_weight=args.first_codebook_weight_multiplier,
            config=monitoring_config,
        )

        main_logger_info(f"  Text evaluation: {args.monitoring.text_evaluation.enabled}")
        main_logger_info(f"  Codebook analysis: {args.monitoring.codebook_analysis.enabled}")
        main_logger_info(f"  Gradient monitoring: {args.monitoring.gradient_monitoring.enabled}")
        main_logger_info("=" * 60)

    # 5.6 Initialize Sample Saver (for 60s segment samples)
    sample_saver = None
    if hasattr(args, "sample_saving") and args.sample_saving is not None:
        # Check both master enable and segment-specific enable
        segment_enabled = args.sample_saving.enabled and getattr(
            args.sample_saving, "save_segment_samples", True
        )
        if segment_enabled:
            main_logger_info("=" * 60)
            main_logger_info("[SEGMENT SAMPLE SAVER ENABLED]")

            sample_saver = SampleSaver(
                mimi=mimi,
                tokenizer=spm,
                run_dir=run_dir,
                text_padding_token_id=unwrapped_model.text_padding_token_id,
                end_of_text_padding_id=unwrapped_model.end_of_text_padding_id,
                audio_offset=unwrapped_model.audio_offset,
                dep_q=unwrapped_model.dep_q,
                has_user_audio=use_stereo_data,  # USER-STREAM or FULL-DUPLEX mode
                sample_rate=args.sample_saving.sample_rate,
                audio_format=args.sample_saving.audio_format,
                max_samples_per_split=args.sample_saving.max_samples_per_split,
                samples_per_save=args.sample_saving.samples_per_save,
                save_audio=args.sample_saving.save_audio,
                save_text=args.sample_saving.save_text,
                debug_audio_consistency=getattr(args.sample_saving, 'debug_audio_consistency', False),
            )

            main_logger_info(f"  Output: samples/{{split}}/step_{{step}}/sample_segment/")
            main_logger_info(f"  Save frequency: every {args.sample_saving.save_freq} steps")
            main_logger_info(f"  Audio: {args.sample_saving.save_audio} ({args.sample_saving.audio_format})")
            main_logger_info(f"  Text: {args.sample_saving.save_text}")
            main_logger_info(f"  Samples per save: {args.sample_saving.samples_per_save}")
            main_logger_info("=" * 60)
        elif args.sample_saving.enabled:
            main_logger_info("[SEGMENT SAMPLE SAVER DISABLED] (save_segment_samples=False)")

    # 5.6.1 Initialize Dialogue Sample Saver (for complete dialogue with GT + Predictions)
    dialogue_saver = None
    if hasattr(args, "sample_saving") and args.sample_saving is not None:
        # Check both master enable and dialogue-specific enable
        dialogue_enabled = args.sample_saving.enabled and getattr(
            args.sample_saving, "save_dialogue_samples", True
        )
        max_dialogues = getattr(args.sample_saving, "max_dialogues_per_split", 5)

        if dialogue_enabled:
            main_logger_info("[DIALOGUE SAMPLE SAVER ENABLED]")
            dialogue_saver = DialogueSampleSaver(
                model=model,  # For inference to generate predictions
                mimi=mimi,
                tokenizer=spm,
                run_dir=run_dir,
                # Use unwrapped_model for attribute access (FSDP compatibility)
                text_padding_token_id=unwrapped_model.text_padding_token_id,
                end_of_text_padding_id=unwrapped_model.end_of_text_padding_id,
                audio_offset=unwrapped_model.audio_offset,
                dep_q=unwrapped_model.dep_q,
                sample_rate=args.sample_saving.sample_rate,
                chunk_duration_sec=args.duration_sec,  # Use same chunk size as training
                max_dialogues_per_split=max_dialogues,
            )
            main_logger_info("  Saves complete dialogues with GT AND Predictions")
            main_logger_info(f"  Output: samples/{{split}}/step_{{step}}/sample_dialogue/")
            main_logger_info(f"  Max dialogues per split: {max_dialogues}")
            main_logger_info("=" * 60)
        elif args.sample_saving.enabled:
            main_logger_info("[DIALOGUE SAMPLE SAVER DISABLED] (save_dialogue_samples=False)")

    # 5.7 Initialize Research Logger
    research_logger = None
    if hasattr(args, "research_logging") and args.research_logging is not None:
        if args.research_logging.enabled:
            main_logger_info("=" * 60)
            main_logger_info("[RESEARCH LOGGER ENABLED]")

            research_logger = ResearchLogger(
                run_dir=run_dir,
                num_codebooks=unwrapped_model.dep_q,
                save_attention_maps=args.research_logging.save_attention_maps,
                attention_freq=args.research_logging.attention_freq,
                attention_samples=args.research_logging.attention_samples,
                save_raw_attention=args.research_logging.save_raw_attention,
                save_loss_curves=args.research_logging.save_loss_curves,
                save_codebook_stats=args.research_logging.save_codebook_stats,
                save_gradient_norms=args.research_logging.save_gradient_norms,
                generate_plots=args.research_logging.generate_plots,
                plot_freq=args.research_logging.plot_freq,
                save_summary=args.research_logging.save_summary,
            )

            # Store training config for summary
            research_logger.set_config(dataclasses.asdict(args))

            main_logger_info(f"  Attention maps: {args.research_logging.save_attention_maps}")
            main_logger_info(f"  Loss curves: {args.research_logging.save_loss_curves}")
            main_logger_info(f"  Codebook stats: {args.research_logging.save_codebook_stats}")
            main_logger_info(f"  Gradient norms: {args.research_logging.save_gradient_norms}")
            main_logger_info(f"  Auto plots: {args.research_logging.generate_plots}")
            main_logger_info("=" * 60)

    # 5.8 Initialize Enhanced Evaluation Orchestrator
    enhanced_evaluator = None
    if hasattr(args, "enhanced_evaluation") and args.enhanced_evaluation is not None:
        main_logger_info("=" * 60)
        main_logger_info("[ENHANCED EVALUATION ENABLED]")

        # Prepare model config for the orchestrator
        model_config = {
            "text_padding_token_id": unwrapped_model.text_padding_token_id,
            "end_of_text_padding_id": unwrapped_model.end_of_text_padding_id,
            "zero_token_id": unwrapped_model.zero_token_id,
            "dep_q": unwrapped_model.dep_q,
            "audio_offset": unwrapped_model.audio_offset,
        }

        enhanced_evaluator = EnhancedEvaluationOrchestrator(
            args=args,
            tokenizer=spm,
            mimi_model=mimi,
            model_config=model_config,
        )

        # Log enabled monitors
        semantic_cfg = args.enhanced_evaluation.semantic
        alignment_cfg = args.enhanced_evaluation.alignment
        dialogue_cfg = args.enhanced_evaluation.dialogue
        audio_cfg = args.enhanced_evaluation.audio_quality

        main_logger_info(f"  Semantic (BLEU): {semantic_cfg.enabled}")
        if semantic_cfg.enabled:
            main_logger_info(f"    compute_bleu: {semantic_cfg.compute_bleu}")
            main_logger_info(f"    compute_semantic: {semantic_cfg.compute_semantic}")
        main_logger_info(f"  Alignment: {alignment_cfg.enabled}")
        main_logger_info(f"  Dialogue: {dialogue_cfg.enabled} (Full-Duplex only)")
        main_logger_info(f"  Audio Quality: {audio_cfg.enabled}")
        if audio_cfg.enabled:
            main_logger_info(f"    compute_pesq: {audio_cfg.compute_pesq}")
            main_logger_info(f"    compute_stoi: {audio_cfg.compute_stoi}")
            main_logger_info(f"    compute_mcd: {audio_cfg.compute_mcd}")
        main_logger_info("=" * 60)

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    # Check if we should use two-rate optimizer (different LRs for TempFormer and DepFormer)
    use_two_rate = (
        args.optim.depformer_lr is not None
        and args.optim.depformer_lr != args.optim.lr
    )

    if use_two_rate:
        main_logger_info("=" * 60)
        main_logger_info("[TWO-RATE OPTIMIZER]")
        main_logger_info(f"  TempFormer LR: {args.optim.lr}")
        main_logger_info(f"  DepFormer LR: {args.optim.depformer_lr}")
        main_logger_info(f"  This follows J-Moshi's approach for better convergence.")
        main_logger_info("=" * 60)

        optimizer = get_two_rate_optimizer(
            model=unwrapped_model,
            tempformer_lr=args.optim.lr,
            depformer_lr=args.optim.depformer_lr,
            weight_decay=args.optim.weight_decay,
            betas=(args.optim.beta1, args.optim.beta2),
            eps=args.optim.eps,
        )
    else:
        main_logger_info(f"[SINGLE-RATE OPTIMIZER] LR: {args.optim.lr}")
        optimizer = AdamW(
            model.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            eps=args.optim.eps,
            weight_decay=args.optim.weight_decay,
            # CRITICAL: foreach=False is required for FSDP resume compatibility
            # The default foreach=True uses multi-tensor operations that require
            # all tensors to be on the same device/dtype, which can break after
            # loading optimizer state from checkpoint in FSDP distributed training.
            foreach=False,
        )

    # Create scheduler based on configuration
    main_logger_info("=" * 60)
    main_logger_info("[SCHEDULER CONFIGURATION]")
    main_logger_info(f"  Type: {args.scheduler.type}")
    main_logger_info(f"  Max steps: {args.max_steps}")

    if args.scheduler.type == "onecycle":
        # For OneCycleLR, use pct_start from optim args for backward compatibility
        main_logger_info(f"  Warmup percentage: {args.optim.pct_start * 100:.1f}%")
        scheduler = get_scheduler(
            scheduler_type="onecycle",
            optimizer=optimizer,
            max_steps=args.max_steps,
            pct_start=args.optim.pct_start,
        )
    else:
        main_logger_info(f"  Warmup steps: {args.scheduler.warmup_steps}")
        main_logger_info(f"  Min LR: {args.scheduler.min_lr}")
        if args.scheduler.type == "cosine_restarts":
            main_logger_info(f"  T_0: {args.scheduler.t_0}, T_mult: {args.scheduler.t_mult}")
        scheduler = get_scheduler(
            scheduler_type=args.scheduler.type,
            optimizer=optimizer,
            max_steps=args.max_steps,
            warmup_steps=args.scheduler.warmup_steps,
            min_lr=args.scheduler.min_lr,
            t_0=args.scheduler.t_0,
            t_mult=args.scheduler.t_mult,
        )

    state = TrainState(args.max_steps)

    # 8. Initialize checkpoint manager
    ckpt_manager = None
    if args.checkpoint.enabled:
        main_logger_info("=" * 60)
        main_logger_info("[CHECKPOINT MANAGER]")
        ckpt_manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=lm_config,
            args=args.checkpoint,
            run_dir=run_dir,
            full_finetuning=args.full_finetuning,
        )
        main_logger_info(f"  Name prefix: {args.checkpoint.name_prefix}")
        main_logger_info(f"  Metric type: {args.checkpoint.metric_type}")
        main_logger_info(f"  Metric best: {args.checkpoint.metric_best}")
        main_logger_info(f"  Save frequency: {args.checkpoint.save_freq}")
        main_logger_info(f"  Max keep: {args.checkpoint.max_keep}")
        main_logger_info(f"  Save optimizer: {args.checkpoint.save_optimizer}")
        main_logger_info(f"  Save scheduler: {args.checkpoint.save_scheduler}")
        main_logger_info(f"  Resume if exist: {args.checkpoint.resume_if_exist}")

        # Check for resume
        if ckpt_manager.should_resume():
            main_logger_info("=" * 60)
            main_logger_info("[RESUMING FROM CHECKPOINT]")
            restored_step, success = ckpt_manager.load_checkpoint()
            if success:
                state = ckpt_manager.state  # Update state reference
                main_logger_info(f"  Resumed from step: {state.step}")
                main_logger_info(f"  Elapsed time: {state.elapsed_time:.1f}s")
                main_logger_info(f"  Seen tokens: {state.n_seen_tokens:,}")
                if state.best_metric is not None:
                    main_logger_info(f"  Best metric: {state.best_metric:.4f} @ step {state.best_step}")
            else:
                main_logger_info("  Resume failed, starting from scratch")

        main_logger_info("=" * 60)

    # Legacy checkpointer (for backward compatibility with do_ckpt flag)
    # This is only used if checkpoint.enabled is False but do_ckpt is True
    legacy_checkpointer = None
    if args.do_ckpt and not args.checkpoint.enabled:
        legacy_checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
        main_logger_info("[LEGACY CHECKPOINTER] Using old-style checkpointing (no resume support)")

    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 10. Training setup
    model.train()
    torch.cuda.empty_cache()

    # Synchronize all ranks before entering training loop
    dist.barrier()

    # Print training start with PrettyLogger
    pretty.print_training_start()

    # Flags for one-time logging
    _padding_logged = False
    _forward_diagnosed = False

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        # Track text data from last batch for sample saving
        last_user_text_alignments = None
        last_moshi_text_raw_list = None
        last_audio_paths = None  # Track audio paths for dialogue saving
        last_speaker_reference_audios = None  # Track speaker references for train sample saving
        last_speaker_embedding = None  # Track speaker embedding for train sample saving
        # Track decoded reference audio from audio_prompt_module (24kHz)
        last_decoded_ref_texts = None
        last_decoded_ref_start_secs = None
        last_decoded_ref_end_secs = None
        last_ref_audio_sample_rate = 16000  # Default to 16kHz (interleaver), updated to 24kHz when from audio_prompt
        # Track prompt length for sample_saver slicing (excludes prompt from GT/Pred alignment)
        last_prompt_length = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes
            original_shape = codes.shape
            last_user_text_alignments = batch.user_text_alignments
            last_moshi_text_raw_list = batch.moshi_text_raw_list
            last_audio_paths = batch.audio_paths  # Track for dialogue saving

            # Pad codes to model's expected codebook count if needed
            # - Stereo modes (USER-STREAM, FULL-DUPLEX): Data has 17 codebooks, no padding
            # - MONOLOGUE mode: Data has 9 codebooks, pad to 17 for moshiko model
            if not use_stereo_data:
                codes = pad_codes_for_model(codes, unwrapped_model.num_codebooks, unwrapped_model.zero_token_id)

            # One-time debug message to verify data shape
            if not _padding_logged:
                if use_stereo_data:
                    mode_name = "USER-STREAM" if enable_user_stream else "FULL-DUPLEX"
                    main_logger_info(f"[{mode_name}] Codes shape: {codes.shape} (17 codebooks from stereo data)")
                else:
                    main_logger_info(
                        f"[MONOLOGUE] Codes padded: {original_shape} -> {codes.shape} "
                        f"(using zero_token_id={unwrapped_model.zero_token_id})"
                    )
                _padding_logged = True

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = unwrapped_model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # Phase 2: Extract speaker embeddings if speaker conditioning is enabled
            speaker_embedding = None
            if speaker_encoder is not None and batch.speaker_reference_audios is not None:
                # Process speaker reference audios for this batch
                ref_audios = batch.speaker_reference_audios
                valid_embeddings = []

                for ref_audio in ref_audios:
                    if ref_audio is not None and ref_audio.numel() > 0:
                        # Move to GPU and encode
                        ref_audio_gpu = ref_audio.cuda()
                        with torch.no_grad():
                            emb = speaker_encoder(ref_audio_gpu)  # [1, D]
                        valid_embeddings.append(emb)
                    else:
                        # Use zero embedding for samples without reference
                        zero_emb = torch.zeros(1, speaker_encoder.config.output_dim, device="cuda")
                        valid_embeddings.append(zero_emb)

                if valid_embeddings:
                    # Stack embeddings into batch [B, D]
                    speaker_embedding = torch.cat(valid_embeddings, dim=0)

                # Track for sample saving
                last_speaker_reference_audios = ref_audios
                last_speaker_embedding = speaker_embedding

            # =================================================================
            # AUDIO PROMPTING FOR TRAINING (Zero-Shot Speaker Adaptation)
            # =================================================================
            # Apply audio/text prompting: prepend reference segment to codes
            # This teaches the model to continue generating in the reference
            # speaker's voice (PersonaPlex style conditioning).
            #
            # The prompt is sampled RANDOMLY from the Moshi stream within the
            # same sequence. This ensures:
            # 1. Reference is from the SAME speaker (Moshi's voice)
            # 2. Model learns: "given this voice sample, continue in this voice"
            #
            # Note: Current implementation samples from the same training
            # segment due to chunking by sphn. For better diversity, consider
            # using longer duration_sec or multiple reference sources.
            # =================================================================
            prompt_mask = None
            prompt_length = 0  # Track prompt length for sample_saver slicing
            train_codes = codes  # Will be replaced if prompting is enabled

            if train_audio_prompt_module is not None:
                # Apply audio prompting with RANDOM sampling
                # Note: We pass exclude_start/end=None to allow sampling from
                # any position in the sequence (same speaker, different segment)
                seq_len = codes.shape[2]
                prompted_codes, prompt_mask, prompt_samples = train_audio_prompt_module(
                    codes,
                    exclude_start=None,   # Allow sampling from any position
                    exclude_end=None,     # (no exclusion for maximum diversity)
                    deterministic=False,  # Random for training
                )
                train_codes = prompted_codes

                # Track prompt length for sample_saver slicing
                if prompt_samples:
                    sample = prompt_samples[0]
                    prompt_length = sample.end_idx - sample.start_idx
                    last_prompt_length = prompt_length  # Persist for sample_saver outside loop

                # One-time logging for audio prompting
                if not _padding_logged and prompt_samples:
                    sample = prompt_samples[0]
                    main_logger_info(
                        f"[AUDIO PROMPTING] Reference segment: "
                        f"frames={sample.start_idx}-{sample.end_idx}, "
                        f"duration={sample.duration_sec:.2f}s, "
                        f"original_codes={list(codes.shape)}, "
                        f"prompted_codes={list(prompted_codes.shape)}"
                    )

                # =============================================================
                # CRITICAL FIX: Decode reference audio from audio_prompt_module
                # =============================================================
                # This ensures Train samples have valid reference audio for saving.
                # Previously, train used batch.speaker_reference_audios which could
                # be None when _sample_speaker_reference() found no valid regions.
                #
                # Now we decode the audio codes from audio_prompt_module's output,
                # matching the Valid behavior in eval.py (lines 345-394).
                # This guarantees reference audio is always available when audio
                # prompting is enabled.
                # =============================================================
                if prompt_samples and mimi is not None:
                    decoded_ref_audios = []
                    decoded_ref_texts = []
                    decoded_ref_start_secs = []
                    decoded_ref_end_secs = []

                    for ps in prompt_samples:
                        if ps.audio_codes is not None:
                            try:
                                # ps.audio_codes is [8, T_prompt] or [K, T_prompt]
                                # Mimi expects [B, K, T] format
                                device = codes.device
                                ref_codes = ps.audio_codes.unsqueeze(0).to(device)
                                ref_codes = ref_codes.clamp(0, 2047)  # Clamp to valid range

                                # Decode using Mimi
                                with torch.no_grad():
                                    ref_audio = mimi.decode(ref_codes)  # [1, 1, T_audio]
                                    # Reshape to [T_audio] for storage
                                    ref_audio = ref_audio.squeeze()  # [T_audio]

                                decoded_ref_audios.append(ref_audio.cpu())

                                # Extract timing from prompt_sample
                                frame_rate = getattr(train_audio_prompt_module.sampler, 'frame_rate', 12.5)
                                start_sec = ps.start_idx / frame_rate
                                end_sec = ps.end_idx / frame_rate
                                decoded_ref_start_secs.append(start_sec)
                                decoded_ref_end_secs.append(end_sec)

                                # Decode text if available
                                if spm is not None and ps.text_tokens is not None:
                                    valid_tokens = [
                                        int(t) for t in ps.text_tokens.tolist()
                                        if int(t) not in {0, 3, 32000} and int(t) >= 0
                                    ]
                                    if valid_tokens:
                                        ref_text = spm.decode(valid_tokens)
                                        decoded_ref_texts.append(ref_text)
                                    else:
                                        decoded_ref_texts.append("")
                                else:
                                    decoded_ref_texts.append("")

                            except Exception as e:
                                if get_rank() == 0:
                                    logger.debug(f"[TRAIN] Failed to decode reference audio: {e}")
                                decoded_ref_audios.append(None)
                                decoded_ref_texts.append("")
                                decoded_ref_start_secs.append(0.0)
                                decoded_ref_end_secs.append(0.0)
                        else:
                            decoded_ref_audios.append(None)
                            decoded_ref_texts.append("")
                            decoded_ref_start_secs.append(0.0)
                            decoded_ref_end_secs.append(0.0)

                    # Update last_speaker_reference_audios with decoded audio
                    # This overrides the interleaver's potentially None values
                    if any(a is not None for a in decoded_ref_audios):
                        last_speaker_reference_audios = decoded_ref_audios
                        # Also store timing and text for metadata
                        last_decoded_ref_texts = decoded_ref_texts
                        last_decoded_ref_start_secs = decoded_ref_start_secs
                        last_decoded_ref_end_secs = decoded_ref_end_secs
                        # CRITICAL: Update sample rate to 24kHz (Mimi output)
                        # This will be used in sample saving to set correct metadata
                        last_ref_audio_sample_rate = 24000

            # forward / backward
            # Speaker conditioning is handled by LMModelWrapper which accepts speaker_embedding
            # When speaker conditioning is enabled, _should_use_modular_backbone() returns True,
            # ensuring LMModelWrapper is used instead of legacy LMModel
            if speaker_embedding is not None and speaker_conditioner is not None:
                output = model(codes=train_codes, condition_tensors=condition_tensors, speaker_embedding=speaker_embedding)
            else:
                output = model(codes=train_codes, condition_tensors=condition_tensors)

            # ONE-TIME DIAGNOSTIC: Check forward pass output for NaN
            if not _forward_diagnosed:
                text_nan = torch.isnan(output.text_logits).sum().item()
                audio_nan = torch.isnan(output.logits).sum().item()

                if text_nan > 0 or audio_nan > 0:
                    main_logger_info(f"[FORWARD PASS] Warning: NaN detected - text={text_nan}, audio={audio_nan}")
                else:
                    main_logger_info(f"[FORWARD PASS] Input: {train_codes.shape}, Output: text={output.text_logits.shape}, audio={output.logits.shape}")

                _forward_diagnosed = True

            # =================================================================
            # CRITICAL: Use train_codes for loss computation
            # =================================================================
            # When audio prompting is enabled, train_codes has reference frames
            # prepended. We MUST use train_codes (not original codes) to match
            # output.text_mask dimensions.
            #
            # The prompt_mask indicates which positions are prompt (True) vs
            # training target (False). Loss is computed on ALL positions, but
            # prompt positions are masked by output.text_mask/output.mask.
            # =================================================================
            loss_codes = train_codes

            text_loss = compute_loss_with_mask(
                output.text_logits,
                loss_codes[:, : unwrapped_model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    unwrapped_model.text_padding_token_id,
                    unwrapped_model.end_of_text_padding_id,
                },
                prompt_mask=prompt_mask,  # Exclude prompt positions from loss
            )

            # Compute audio loss with per-speaker breakdown for user stream mode
            audio_codes = loss_codes[:, unwrapped_model.audio_offset : unwrapped_model.audio_offset + unwrapped_model.dep_q]

            if enable_user_stream and unwrapped_model.dep_q == 16:
                # Full duplex mode: separate moshi and user audio losses
                audio_loss_result: AudioLossResult = compute_audio_loss_per_speaker(
                    output.logits,
                    audio_codes,
                    output.mask,
                    dep_q=unwrapped_model.dep_q,
                    semantic_weight=args.first_codebook_weight_multiplier,
                    acoustic_weight=1.0,
                    user_semantic_weight=args.user_semantic_weight,
                    user_acoustic_weight=args.user_acoustic_weight,
                    prompt_mask=prompt_mask,  # Exclude prompt positions from loss
                )
                # Apply user stream loss ratio
                audio_loss = (
                    audio_loss_result.moshi_total_loss +
                    audio_loss_result.user_total_loss * args.user_stream_loss_ratio
                ) / (1.0 + args.user_stream_loss_ratio)

                # Store for logging (will be used at log_freq intervals)
                if not hasattr(state, '_audio_loss_result'):
                    state._audio_loss_result = audio_loss_result
                else:
                    state._audio_loss_result = audio_loss_result
            else:
                # Standard mono mode
                audio_loss = compute_loss_with_mask(
                    output.logits,
                    audio_codes,
                    output.mask,
                    mode="audio",
                    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                    prompt_mask=prompt_mask,  # Exclude prompt positions from loss
                )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    # FSDP shards parameters across ranks, so some parameters
                    # may not have gradients on this rank. Skip the assertion
                    # and only divide gradients that exist.
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # Advanced monitoring: Check gradient health before optimizer step
        gradient_health = None
        if advanced_monitor is not None and args.monitoring.gradient_monitoring.enabled:
            if state.step % args.monitoring.gradient_monitoring.log_freq == 0:
                gradient_health = advanced_monitor.check_gradients(unwrapped_model)

        # optimizer step
        optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        # Get current learning rate(s) - supports both single and two-rate optimizers
        current_lrs = get_current_lr(scheduler)
        if use_two_rate:
            last_lr = current_lrs.get("tempformer", scheduler.get_last_lr()[0])
        else:
            last_lr = current_lrs.get("lr", scheduler.get_last_lr()[0])
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # Create fresh eval data loader for each evaluation
            # (generators can only be iterated once, so we need a new one each time)
            eval_data_loader = create_eval_data_loader()

            # write perplexity to state
            # Phase 2: Pass speaker_encoder and audio_prompt_module for speaker conditioning
            # Returns EvalReturnData with both original_codes and prompted_codes
            eval_data: EvalReturnData = evaluate(
                model,
                eval_data_loader,
                state,
                args,
                speaker_encoder=speaker_encoder,
                audio_prompt_module=audio_prompt_module,
                mimi=mimi,  # For decoding reference audio from prompt codes
                text_tokenizer=spm,  # For decoding reference text from prompt tokens
            )
            # Extract fields from EvalReturnData
            eval_original_codes = eval_data.original_codes  # For sample saving
            eval_prompted_codes = eval_data.prompted_codes  # For metric computation
            eval_output = eval_data.output
            eval_user_text_alignments = eval_data.user_text_alignments
            eval_moshi_text_raw_list = eval_data.moshi_text_raw_list
            eval_audio_paths = eval_data.audio_paths
            eval_speaker_cond_info = eval_data.speaker_conditioning_info

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            # Use PrettyLogger for enhanced evaluation results display
            pretty.print_eval_results(
                step=state.step,
                eval_loss=state.this_eval_loss,
                text_loss=state.this_text_loss,
                audio_loss=state.this_audio_loss,
                perplexity=state.this_eval_perplexity,
            )
            eval_logger.log(eval_logs, step=state.step)

            # =========================================================================
            # ENHANCED EVALUATION: BLEU, Alignment, Dialogue, Audio Quality
            # =========================================================================
            # CRITICAL: Use eval_prompted_codes (not eval_original_codes) for metrics
            # because eval_output.mask shape matches prompted_codes dimensions
            if enhanced_evaluator is not None and eval_prompted_codes is not None and eval_output is not None:
                with torch.no_grad():
                    # Create a batch-like object for the orchestrator
                    # Use prompted_codes to match output.mask dimensions
                    eval_batch = SimpleNamespace(
                        codes=eval_prompted_codes,
                        user_text_alignments=eval_user_text_alignments,
                        moshi_text_raw_list=eval_moshi_text_raw_list,
                    )

                    # Run enhanced evaluation
                    enhanced_result = enhanced_evaluator.evaluate_batch(
                        eval_batch,
                        eval_output,
                        model=unwrapped_model,
                    )

                    # Get all metrics and log them
                    enhanced_metrics = enhanced_evaluator.get_all_metrics()

                    # Add "enhanced/" prefix to distinguish from standard metrics
                    prefixed_metrics = {
                        f"enhanced/{k}": v for k, v in enhanced_metrics.items()
                    }
                    eval_logger.log(prefixed_metrics, step=state.step)

                    # Log summary message
                    summary = enhanced_evaluator.format_summary_message(state.step)
                    main_logger_info(summary)

                    # Reset for next evaluation
                    enhanced_evaluator.reset_all()

            # Save validation samples if sample_saver is enabled
            # Saves both Ground Truth and Predictions for comparison
            # CRITICAL: Use eval_original_codes (not prompted) for sample saving
            # This ensures GT audio is from original data, not including prompt prefix
            # CRITICAL: Only rank 0 saves samples to avoid race conditions in multinode
            if sample_saver is not None and eval_original_codes is not None and eval_output is not None and get_rank() == 0:
                with torch.no_grad():
                    # =================================================================
                    # CRITICAL FIX: Slice output logits to exclude prompt region
                    # =================================================================
                    # When audio prompting is enabled, output logits have shape
                    # [B, K, T_main + T_prompt, V]. We need to slice to match
                    # eval_original_codes shape [B, K, T_main] for correct GT/Pred alignment.
                    # =================================================================
                    eval_prompt_length = 0
                    if eval_prompted_codes is not None and eval_original_codes is not None:
                        eval_prompt_length = eval_prompted_codes.size(2) - eval_original_codes.size(2)
                        eval_prompt_length = max(0, eval_prompt_length)  # Ensure non-negative

                    if eval_prompt_length > 0:
                        sliced_text_logits = eval_output.text_logits[:, :, eval_prompt_length:]
                        sliced_audio_logits = eval_output.logits[:, :, eval_prompt_length:]
                    else:
                        sliced_text_logits = eval_output.text_logits
                        sliced_audio_logits = eval_output.logits

                    save_result = sample_saver.save_samples(
                        codes=eval_original_codes,  # Use original codes for GT audio
                        text_logits=sliced_text_logits,
                        audio_logits=sliced_audio_logits,
                        step=state.step,
                        split="valid",
                        user_text_alignments=eval_user_text_alignments,
                        moshi_text_raw_list=eval_moshi_text_raw_list,
                        speaker_conditioning_info=eval_speaker_cond_info,  # Phase 1: Speaker conditioning metadata
                        prompt_length=eval_prompt_length,  # For shape validation
                    )
                    pretty.print_sample_saved(
                        state.step, "valid",
                        save_result.num_samples,
                        len(save_result.dialogue_paths),
                    )

            # Save complete validation dialogues (GT + Predictions) for evaluation
            # CRITICAL: Only rank 0 saves dialogues to avoid race conditions in multinode
            if dialogue_saver is not None and eval_audio_paths is not None and get_rank() == 0:
                main_speaker_label = interleaver_cfg.main_speaker_label
                dialogue_results = dialogue_saver.save_dialogue_from_batch(
                    batch_paths=eval_audio_paths,
                    step=state.step,
                    main_speaker_label=main_speaker_label,
                    max_per_batch=1,
                    split="valid",
                )
                for result in dialogue_results:
                    if result.success:
                        main_logger_info(
                            f"[DIALOGUE SAVED] [VALID] {result.path} -> {result.output_dir} "
                            f"({result.duration_sec:.1f}s, moshi={result.moshi_word_count}w, "
                            f"user={result.user_word_count}w)"
                        )

            # CRITICAL: Barrier after all eval operations to ensure all ranks sync before checkpoint
            # This prevents race conditions where rank 0 starts checkpoint saving while others are still busy
            dist.barrier()

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            # Pass learning rate(s) to log - use dict for two-rate optimizer
            lr_for_log = current_lrs if use_two_rate else last_lr
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                lr_for_log,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )

            # Add user stream loss metrics to train_logs if available
            if enable_user_stream and hasattr(state, '_audio_loss_result'):
                audio_loss_result = state._audio_loss_result
                train_logs["train/moshi_semantic_loss"] = audio_loss_result.moshi_semantic_loss.item()
                train_logs["train/moshi_acoustic_loss"] = audio_loss_result.moshi_acoustic_loss.item()
                train_logs["train/moshi_total_loss"] = audio_loss_result.moshi_total_loss.item()
                if audio_loss_result.user_total_loss is not None:
                    train_logs["train/user_semantic_loss"] = audio_loss_result.user_semantic_loss.item()
                    train_logs["train/user_acoustic_loss"] = audio_loss_result.user_acoustic_loss.item()
                    train_logs["train/user_total_loss"] = audio_loss_result.user_total_loss.item()

            # Use PrettyLogger for enhanced step progress display
            gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
            samples_per_sec = train_logs.get("train/tok_per_sec", 0) / 1000 if "train/tok_per_sec" in train_logs else 0

            # Enhanced logging for user stream mode
            if enable_user_stream and hasattr(state, '_audio_loss_result'):
                audio_loss_result = state._audio_loss_result
                pretty.print_step_progress(
                    step=state.step,
                    max_steps=args.max_steps,
                    loss=avg_loss,
                    text_loss=state.this_text_loss,
                    audio_loss=state.this_audio_loss,
                    lr=last_lr,
                    grad_norm=train_logs.get("train/grad_norm", 0),
                    samples_per_sec=samples_per_sec,
                    gpu_memory_gb=gpu_mem_gb,
                    moshi_loss=audio_loss_result.moshi_total_loss.item() if audio_loss_result else None,
                    user_loss=audio_loss_result.user_total_loss.item() if audio_loss_result and audio_loss_result.user_total_loss is not None else None,
                )

                # Detailed Full Duplex loss breakdown at lower frequency
                if state.step % (args.log_freq * 10) == 0 and audio_loss_result is not None:
                    pretty.print_user_stream_losses(
                        step=state.step,
                        moshi_semantic=audio_loss_result.moshi_semantic_loss.item(),
                        moshi_acoustic=audio_loss_result.moshi_acoustic_loss.item(),
                        user_semantic=audio_loss_result.user_semantic_loss.item() if audio_loss_result.user_semantic_loss is not None else 0.0,
                        user_acoustic=audio_loss_result.user_acoustic_loss.item() if audio_loss_result.user_acoustic_loss is not None else 0.0,
                    )
            else:
                pretty.print_step_progress(
                    step=state.step,
                    max_steps=args.max_steps,
                    loss=avg_loss,
                    text_loss=state.this_text_loss,
                    audio_loss=state.this_audio_loss,
                    lr=last_lr,
                    grad_norm=train_logs.get("train/grad_norm", 0),
                    samples_per_sec=samples_per_sec,
                    gpu_memory_gb=gpu_mem_gb,
                )
            metrics_logger.log(train_logs, step=state.step)

            # Advanced monitoring: Add gradient health metrics
            if gradient_health is not None:
                gradient_metrics = {
                    "gradient/norm": gradient_health.grad_norm,
                    "gradient/has_nan": int(gradient_health.has_nan),
                    "gradient/has_inf": int(gradient_health.has_inf),
                    "gradient/is_exploding": int(gradient_health.is_exploding),
                    "gradient/is_vanishing": int(gradient_health.is_vanishing),
                }
                metrics_logger.log(gradient_metrics, step=state.step)

                # Display gradient health with visual indicators (only on issues)
                if gradient_health.has_nan or gradient_health.has_inf or gradient_health.is_exploding or gradient_health.is_vanishing:
                    pretty.print_gradient_health(
                        step=state.step,
                        grad_norm=gradient_health.grad_norm,
                        has_nan=gradient_health.has_nan,
                        has_inf=gradient_health.has_inf,
                        is_exploding=gradient_health.is_exploding,
                        is_vanishing=gradient_health.is_vanishing,
                    )

        # Advanced monitoring: Text evaluation and codebook analysis (periodic)
        if advanced_monitor is not None:
            # Text evaluation at specified frequency
            text_eval_freq = getattr(
                args.monitoring.text_evaluation, "eval_freq",
                args.eval_freq if args.eval_freq > 0 else 500
            )
            if args.monitoring.text_evaluation.enabled and state.step % text_eval_freq == 0:
                with torch.no_grad():
                    text_eval_result = advanced_monitor.evaluate_text(
                        output.text_logits,
                        codes[:, : unwrapped_model.audio_offset],
                        output.text_mask,
                    )
                    if text_eval_result is not None:
                        text_metrics = {
                            "text_eval/wer": text_eval_result.wer,
                            "text_eval/cer": text_eval_result.cer,
                        }
                        metrics_logger.log(text_metrics, step=state.step)

                        # Display sample predictions with enhanced formatting
                        if args.monitoring.text_evaluation.log_predictions and text_eval_result.samples:
                            pretty.print_text_predictions(
                                step=state.step,
                                samples=text_eval_result.samples,
                                wer=text_eval_result.wer,
                                cer=text_eval_result.cer,
                                max_display=args.monitoring.text_evaluation.max_prediction_samples,
                            )

            # Codebook analysis at specified frequency
            codebook_log_freq = getattr(
                args.monitoring.codebook_analysis, "log_freq", 100
            )
            if args.monitoring.codebook_analysis.enabled and state.step % codebook_log_freq == 0:
                with torch.no_grad():
                    # CRITICAL: Use train_codes (which may be prompted_codes) to match output.mask dimensions
                    # When audio prompting is enabled, output.mask shape is [B, dep_q, T + T_prompt]
                    # but codes shape is [B, K, T] (without prompt).
                    # train_codes is already set to prompted_codes when audio prompting is active.
                    analysis_codes = train_codes[:, unwrapped_model.audio_offset : unwrapped_model.audio_offset + unwrapped_model.dep_q]

                    # Defensive check: ensure shapes match
                    if analysis_codes.shape[-1] != output.mask.shape[-1]:
                        if get_rank() == 0:
                            logger.debug(
                                f"[CODEBOOK] Shape mismatch: codes T={analysis_codes.shape[-1]} vs mask T={output.mask.shape[-1]}, skipping analysis"
                            )
                        codebook_result = None
                    else:
                        codebook_result = advanced_monitor.analyze_codebooks(
                            output.logits,
                            analysis_codes,
                            output.mask,
                        )
                    if codebook_result is not None:
                        codebook_metrics = {
                            "codebook/semantic_loss": codebook_result.semantic_loss,
                            "codebook/acoustic_loss": codebook_result.acoustic_loss,
                        }
                        for i, loss_val in enumerate(codebook_result.losses):
                            codebook_metrics[f"codebook/cb{i}_loss"] = loss_val
                        if codebook_result.entropy is not None:
                            for i, ent in enumerate(codebook_result.entropy):
                                codebook_metrics[f"codebook/cb{i}_entropy"] = ent
                        metrics_logger.log(codebook_metrics, step=state.step)

                        # Display codebook analysis with visual bar chart
                        pretty.print_codebook_analysis(
                            step=state.step,
                            losses=codebook_result.losses,
                            entropy=codebook_result.entropy,
                            semantic_loss=codebook_result.semantic_loss,
                            acoustic_loss=codebook_result.acoustic_loss,
                        )

        # Sample saving (periodic)
        # Saves both Ground Truth and Predictions for comparison:
        #   - gt_moshi.wav: What the model should generate
        #   - pred_moshi.wav: What the model actually generates
        #   - gt_dialogue.wav: Stereo (L=Moshi GT, R=User GT)
        #   - pred_dialogue.wav: Stereo (L=Moshi Pred, R=User GT)
        #   - text.json: Ground truth vs predicted text
        #   - reference.wav: Speaker reference audio (if speaker conditioning enabled)
        #   - speaker_metadata.json: Speaker conditioning metadata
        # CRITICAL: Only rank 0 saves samples to avoid race conditions in multinode
        if sample_saver is not None and get_rank() == 0:
            save_freq = args.sample_saving.save_freq
            if state.step % save_freq == 0:
                with torch.no_grad():
                    # Build speaker conditioning info for train samples
                    train_speaker_cond_info = None
                    if (speaker_conditioning_config is not None and
                        speaker_conditioning_config.get("enabled", False) and
                        last_speaker_reference_audios is not None):
                        # =============================================================
                        # CRITICAL: Use audio_prompt_module decoded audio (24kHz) when
                        # available, otherwise fall back to interleaver audio (16kHz)
                        # =============================================================
                        # The audio_prompt_module decoding is more reliable because it
                        # always succeeds when prompting is enabled. The interleaver's
                        # _sample_speaker_reference() may return None when no valid
                        # regions are found.
                        # =============================================================
                        ref_target_sample_rate = last_ref_audio_sample_rate

                        # Create EvalSpeakerConditioningInfo for train sample saving
                        train_speaker_cond_info = EvalSpeakerConditioningInfo(
                            enabled=True,
                            method=getattr(args.speaker, 'method', 'encoder'),
                            deterministic=False,  # Train uses random sampling
                            sampling_strategy="random",  # Train sampling is random
                            reference_audio_sample_rate=ref_target_sample_rate,
                        )
                        # Set speaker embedding if available
                        if last_speaker_embedding is not None:
                            train_speaker_cond_info.speaker_embedding = last_speaker_embedding

                        # =============================================================
                        # CRITICAL FIX: Store BATCH-LEVEL info for per-sample metadata
                        # =============================================================
                        # Previously, only the first sample's info was stored, causing
                        # all speaker_metadata.json files to be identical.
                        # Now we store lists for each batch item.
                        # =============================================================

                        # Set reference texts as LIST (per-batch-item)
                        if last_decoded_ref_texts is not None:
                            train_speaker_cond_info.reference_texts = last_decoded_ref_texts
                            # Also set legacy single field (for compatibility)
                            if last_decoded_ref_texts:
                                train_speaker_cond_info.reference_text = last_decoded_ref_texts[0]
                        elif batch.speaker_reference_texts is not None:
                            train_speaker_cond_info.reference_texts = batch.speaker_reference_texts
                            if batch.speaker_reference_texts:
                                train_speaker_cond_info.reference_text = batch.speaker_reference_texts[0]

                        # Set reference timing as LISTS (per-batch-item)
                        if last_decoded_ref_start_secs is not None and last_decoded_ref_start_secs:
                            train_speaker_cond_info.reference_start_secs = last_decoded_ref_start_secs
                            train_speaker_cond_info.reference_end_secs = last_decoded_ref_end_secs
                            # Legacy single fields
                            train_speaker_cond_info.reference_start_sec = last_decoded_ref_start_secs[0]
                            train_speaker_cond_info.reference_end_sec = last_decoded_ref_end_secs[0]
                        elif batch.speaker_reference_start_secs is not None:
                            train_speaker_cond_info.reference_start_secs = batch.speaker_reference_start_secs
                            train_speaker_cond_info.reference_end_secs = batch.speaker_reference_end_secs
                            train_speaker_cond_info.reference_start_sec = batch.speaker_reference_start_secs[0]
                            train_speaker_cond_info.reference_end_sec = batch.speaker_reference_end_secs[0]

                        # Track source files as LIST (per-batch-item)
                        if batch.audio_paths is not None and batch.audio_paths:
                            train_speaker_cond_info.source_files = batch.audio_paths
                            train_speaker_cond_info.source_file = batch.audio_paths[0]

                        # Set reference audio from last_speaker_reference_audios
                        # (now populated by audio_prompt_module decode at 24kHz)
                        # Stack reference audios into [B, T] format
                        ref_audio_list = []
                        max_len = 0
                        total_samples = 0
                        for ref_audio in last_speaker_reference_audios:
                            if ref_audio is not None and ref_audio.numel() > 0:
                                # ref_audio is [1, T] from dataset, squeeze to [T]
                                if ref_audio.dim() == 2:
                                    ref_audio = ref_audio.squeeze(0)
                                max_len = max(max_len, ref_audio.size(-1))
                                total_samples += ref_audio.size(-1)
                                ref_audio_list.append(ref_audio)
                            else:
                                ref_audio_list.append(None)

                        # Pad and stack if we have valid reference audios
                        if max_len > 0 and any(r is not None for r in ref_audio_list):
                            padded_refs = []
                            # Determine target device from codes tensor (which is on CUDA)
                            target_device = codes.device
                            for ref_audio in ref_audio_list:
                                if ref_audio is not None:
                                    # Move to target device first (CPU -> CUDA)
                                    ref_audio = ref_audio.to(target_device)
                                    # Pad to max_len
                                    if ref_audio.size(-1) < max_len:
                                        pad_size = max_len - ref_audio.size(-1)
                                        ref_audio = torch.nn.functional.pad(ref_audio, (0, pad_size))
                                    padded_refs.append(ref_audio)
                                else:
                                    # Zero tensor for missing reference (on same device)
                                    padded_refs.append(torch.zeros(max_len, device=target_device))

                            # Stack to [B, T]
                            train_speaker_cond_info.reference_audio = torch.stack(padded_refs, dim=0)
                            # Update metadata with actual values
                            train_speaker_cond_info.reference_num_frames = max_len
                            train_speaker_cond_info.reference_duration_sec = max_len / ref_target_sample_rate

                    # =================================================================
                    # CRITICAL FIX: Slice output logits to exclude prompt region
                    # =================================================================
                    # When audio prompting is enabled, output logits have shape
                    # [B, K, T_main + T_prompt, V]. We need to slice to match
                    # codes shape [B, K, T_main] for correct GT/Pred alignment.
                    # =================================================================
                    if last_prompt_length > 0:
                        sliced_text_logits = output.text_logits[:, :, last_prompt_length:]
                        sliced_audio_logits = output.logits[:, :, last_prompt_length:]
                    else:
                        sliced_text_logits = output.text_logits
                        sliced_audio_logits = output.logits

                    save_result = sample_saver.save_samples(
                        codes=codes,
                        text_logits=sliced_text_logits,
                        audio_logits=sliced_audio_logits,
                        step=state.step,
                        split="train",
                        user_text_alignments=last_user_text_alignments,
                        moshi_text_raw_list=last_moshi_text_raw_list,
                        speaker_conditioning_info=train_speaker_cond_info,
                        prompt_length=last_prompt_length,  # For shape validation
                    )
                    pretty.print_sample_saved(
                        state.step, "train",
                        save_result.num_samples,
                        len(save_result.dialogue_paths),
                    )

        # =====================================================================
        # Complete Dialogue Saving (GT + Predictions via Model Inference)
        # =====================================================================
        # This saves ENTIRE dialogues with BOTH Ground Truth AND Predictions.
        # The model forward pass is run on encoded dialogue to generate predictions.
        #
        # Output: samples/{split}/step_{step}/sample_dialogue/
        #   - sample_XX_gt_dialogue.wav: GT stereo (L=Moshi, R=User)
        #   - sample_XX_gt_moshi.wav: GT Moshi audio
        #   - sample_XX_pred_dialogue.wav: Pred stereo (L=Pred Moshi, R=GT User)
        #   - sample_XX_pred_moshi.wav: Predicted Moshi audio
        #   - sample_XX_text.json: Text comparison (moshi GT/pred, user GT)
        # CRITICAL: Only rank 0 saves dialogues to avoid race conditions in multinode
        if dialogue_saver is not None and last_audio_paths is not None and get_rank() == 0:
            save_freq = args.sample_saving.save_freq
            if state.step % save_freq == 0:
                # Save complete dialogues from original source files
                main_speaker_label = interleaver_cfg.main_speaker_label
                dialogue_results = dialogue_saver.save_dialogue_from_batch(
                    batch_paths=last_audio_paths,
                    step=state.step,
                    main_speaker_label=main_speaker_label,
                    max_per_batch=1,  # Save 1 dialogue per batch to avoid duplicates
                    split="train",
                )
                for result in dialogue_results:
                    if result.success:
                        main_logger_info(
                            f"[DIALOGUE SAVED] [TRAIN] {result.path} -> {result.output_dir} "
                            f"({result.duration_sec:.1f}s, moshi={result.moshi_word_count}w, "
                            f"user={result.user_word_count}w)"
                        )

        # Research logging (periodic)
        if research_logger is not None:
            # Log loss data
            research_logger.log_loss(
                step=state.step,
                loss=avg_loss,
                text_loss=text_loss.item() if isinstance(text_loss, torch.Tensor) else None,
                audio_loss=audio_loss.item() if isinstance(audio_loss, torch.Tensor) else None,
                codebook_losses=None,  # Populated by codebook analysis if enabled
                lr=last_lr,
                split="train",
            )

            # Log gradient data
            if gradient_health is not None:
                research_logger.log_gradient(
                    step=state.step,
                    grad_norm=gradient_health.grad_norm,
                    has_nan=gradient_health.has_nan,
                    has_inf=gradient_health.has_inf,
                )

            # Periodic plot generation and data saving
            research_logger.periodic_save(state.step)

        # CRITICAL: Barrier before checkpoint to ensure all ranks finish sample saving
        # This prevents race conditions where some ranks start checkpoint collective
        # operations while others are still doing I/O (sample/research logging)
        dist.barrier()

        # Save checkpoint using new CheckpointManager or legacy Checkpointer
        if ckpt_manager is not None:
            # New checkpoint manager: saves based on metric value
            # Determine the metric value to use for checkpoint naming
            metric_value = state.get_metric(args.checkpoint.metric_type)
            if metric_value is None:
                # Fallback: use current loss if metric not available
                metric_value = avg_loss

            # Store train loss in state for metric tracking
            state.this_train_loss = avg_loss

            # Check if we should save at this step
            should_save = (
                (args.checkpoint.save_freq > 0 and state.step % args.checkpoint.save_freq == 0)
                or is_last_step
            )
            if should_save:
                ckpt_path = ckpt_manager.save_checkpoint(
                    metric_value=metric_value,
                    force_save=is_last_step,
                )
                if ckpt_path is not None:
                    pretty.print_checkpoint_saved(state.step, str(run_dir / "checkpoints"))

        elif legacy_checkpointer is not None:
            # Legacy checkpointer (backward compatibility)
            if args.do_ckpt and (
                (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
            ):
                legacy_checkpointer.save_checkpoint(
                    save_only_lora=not args.full_finetuning and args.save_adapters,
                    dtype=param_dtype,
                )
                pretty.print_checkpoint_saved(state.step, str(run_dir / "checkpoints"))

    # Calculate total training time
    total_time = time.time() - pretty.start_time if pretty.start_time else 0

    # Print training completion summary
    pretty.print_training_complete(
        total_steps=state.step,
        final_loss=avg_loss,
        best_loss=state.best_eval_loss if hasattr(state, 'best_eval_loss') else avg_loss,
        best_step=state.best_step if hasattr(state, 'best_step') else state.step,
        total_time_seconds=total_time,
    )

    # Finalize research logger
    if research_logger is not None:
        pretty.print_info("Finalizing research logger...")
        research_logger.finalize(state.step)


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
