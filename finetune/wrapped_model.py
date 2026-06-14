"""
Model wrapping utilities for distributed training.

This module provides functions to wrap models for distributed training using either:
- FSDP (FullyShardedDataParallel): Shards model parameters across GPUs
- DDP (DistributedDataParallel): Replicates full model on each GPU

FSDP is more memory efficient but more complex.
DDP is simpler and better supported with MPI launchers.

LoRA Support Note:
    The moshi library handles LoRA parameters differently from typical models.
    LoRA params (lora, lora_rank, lora_scaling) should NOT be passed directly
    to LMModel.__init__. Instead, the moshi library:

    1. Creates the base model WITHOUT LoRA parameters
    2. If LoRA is enabled, calls replace_all_linear_with_lora() AFTER model creation

    For LoRA finetuning, use checkpointer_info.get_moshi() which handles this
    correctly. For dynamic extension mode (Full Duplex), LoRA should be applied
    after weight loading and extension, using:

        from moshi.models.loaders import get_lora_moshi
        model = get_lora_moshi(model, lora_rank, lora_scaling, ...)

    Currently, full finetuning mode is fully supported. LoRA support in dynamic
    extension mode requires additional implementation.
"""

import functools
import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from moshi.models.lm import LMModel
from moshi.models.loaders import CheckpointInfo, _is_safetensors, _lm_kwargs
from moshi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size

# Import modular backbone system
from .backbone import (
    BackboneFactory,
    MoshiBackbone,
    LMModelWrapper,
    create_lm_model_wrapper,
    wrap_existing_lm_model,
)
from .backbone.config import UnifiedBackboneConfig

# Import model extension utilities for Full Duplex support
try:
    from tools.model_utils import (
        extend_moshi_modules_for_user_stream,
        get_model_architecture_info,
    )
    MODEL_UTILS_AVAILABLE = True
except ImportError:
    MODEL_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _should_use_modular_backbone(args: TrainArgs) -> bool:
    """
    Check if we should use the modular backbone system.

    Returns True for:
    - "hf_lm" or "custom" backbone types
    - "moshi" backbone when speaker conditioning is enabled (requires LMModelWrapper)

    Returns False for:
    - "moshi" backbone without speaker conditioning (legacy LMModel path)

    This enables gradual migration to the modular backbone system while
    maintaining backward compatibility with existing Moshi training.

    Args:
        args: Training arguments with backbone configuration

    Returns:
        True if modular backbone system should be used
    """
    backbone_type = args.backbone.type.lower()

    # Check if speaker conditioning is enabled
    speaker_conditioning_enabled = (
        hasattr(args, "speaker")
        and args.speaker is not None
        and args.speaker.enabled
    )

    if backbone_type == "moshi":
        if speaker_conditioning_enabled:
            # Speaker conditioning requires LMModelWrapper for sum_condition support
            # Original LMModel.forward() doesn't accept sum_condition parameter
            logger.info(
                "[BACKBONE] Using modular backbone system for Moshi (speaker conditioning enabled)"
            )
            return True
        else:
            # Use legacy LMModel loading path for backward compatibility
            return False
    elif backbone_type == "hf_lm":
        # HFLM backbone requires modular system
        logger.info("[BACKBONE] Using modular backbone system for HFLM")
        return True
    elif backbone_type == "custom":
        # Custom backbones use modular system
        logger.info("[BACKBONE] Using modular backbone system for custom backbone")
        return True
    else:
        logger.warning(
            f"[BACKBONE] Unknown backbone type '{backbone_type}', "
            "falling back to legacy LMModel path"
        )
        return False


def _get_modular_backbone_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> Union[FullyShardedDataParallel, DistributedDataParallel, LMModelWrapper]:
    """
    Create and return a distributed model using the modular backbone system.

    This function implements Phase 3 of the modular backbone architecture:
    1. Load base Moshi model to extract shared components (embeddings, depformer)
    2. Create LMModelWrapper with the specified backbone (HFLM, custom, etc.)
    3. Handle distributed wrapping (FSDP or DDP)

    Architecture Flow:
        Base LMModel → Extract Components → LMModelWrapper + Backbone → FSDP/DDP

    Args:
        args: Training arguments with backbone and distributed settings
        checkpointer_info: Checkpoint information for loading base weights

    Returns:
        Distributed-wrapped LMModelWrapper (FSDP, DDP, or plain for single GPU)

    Raises:
        ValueError: If backbone configuration is invalid
        RuntimeError: If model creation or weight loading fails
    """
    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32
    else:
        param_dtype = torch.bfloat16

    device = torch.device("cuda")

    main_logger_info("=" * 70)
    main_logger_info("[MODULAR BACKBONE] Initializing LMModelWrapper")
    main_logger_info(f"  Backbone type: {args.backbone.type}")
    main_logger_info(f"  Distributed backend: {args.distributed_backend}")
    main_logger_info(f"  Param dtype: {param_dtype}")
    main_logger_info("=" * 70)

    # =========================================================================
    # Step 1: Load base Moshi model to extract shared components
    # =========================================================================
    main_logger_info("[Step 1/4] Loading base Moshi model for shared components...")

    # Load original Moshi model with weights
    # This gives us the trained embeddings, depformer, and linear layers
    base_model = checkpointer_info.get_moshi(
        device="cpu",  # Load to CPU first, then move components
        dtype=param_dtype,
        lm_kwargs_overrides={
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    )

    main_logger_info(
        f"  Base model loaded: n_q={base_model.n_q}, dep_q={base_model.dep_q}, "
        f"dim={base_model.dim}"
    )

    # =========================================================================
    # Step 2: Create LMModelWrapper with modular backbone
    # =========================================================================
    main_logger_info("[Step 2/4] Creating LMModelWrapper with modular backbone...")

    try:
        lm_wrapper = LMModelWrapper.from_config(
            config=args.backbone,
            base_lm_model=base_model,
            device=device,
            dtype=param_dtype,
        )
    except Exception as e:
        logger.error(f"[MODULAR BACKBONE] Failed to create LMModelWrapper: {e}")
        raise RuntimeError(
            f"Failed to create LMModelWrapper with backbone '{args.backbone.type}'.\n"
            f"Error: {e}\n"
            f"\n"
            f"Please check:\n"
            f"  1. Backbone configuration in YAML\n"
            f"  2. Model path exists (for HFLM: backbone.hf_lm.model_path)\n"
            f"  3. Dimension adapter settings match backbone dimensions"
        ) from e

    main_logger_info(
        f"  LMModelWrapper created: backbone_dim={lm_wrapper.backbone_dim}, "
        f"moshi_dim={lm_wrapper.moshi_dim}, adapter_enabled={lm_wrapper.adapter.is_enabled}"
    )

    # =========================================================================
    # Step 3: Set requires_grad based on training configuration
    # =========================================================================
    main_logger_info("[Step 3/4] Configuring trainable parameters...")

    if args.lora.enable and not args.full_finetuning:
        # LoRA mode: Only train LoRA parameters and optionally embeddings
        # Note: LoRA for modular backbone needs to be applied to the backbone
        logger.warning(
            "[MODULAR BACKBONE] LoRA mode with modular backbone is experimental. "
            "Consider using full finetuning for best results."
        )
        for name, param in lm_wrapper.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Full finetuning: Train all parameters
        for param in lm_wrapper.parameters():
            param.requires_grad = True

    # Count trainable parameters
    total_params = sum(p.numel() for p in lm_wrapper.parameters())
    trainable_params = sum(p.numel() for p in lm_wrapper.parameters() if p.requires_grad)

    main_logger_info(
        f"  Trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )

    # =========================================================================
    # Step 4: Wrap with distributed backend
    # =========================================================================
    main_logger_info("[Step 4/4] Applying distributed wrapper...")

    if get_world_size() == 1:
        main_logger_info("  Single GPU mode - no distributed wrapper needed")
        main_logger_info("=" * 70)
        return lm_wrapper

    torch.distributed.barrier()

    if args.distributed_backend == "fsdp":
        main_logger_info(f"  Wrapping with FSDP over {get_world_size()} GPUs...")

        # Custom FSDP wrap policy for LMModelWrapper
        # Wrap backbone transformer layers and depformer layers separately
        # Include both Moshi's StreamingTransformerLayer and the HF LM decoder layer

        # Dynamically collect transformer layer classes
        transformer_layer_classes = [StreamingTransformerLayer]

        # Try to import HFLM decoder layer class for FSDP wrapping
        try:
            # Check if backbone is HFLM and get its layer class
            if hasattr(lm_wrapper, 'backbone') and hasattr(lm_wrapper.backbone, 'model'):
                hf_model = lm_wrapper.backbone.model
                if hasattr(hf_model, 'layers') and len(hf_model.layers) > 0:
                    layer_class = type(hf_model.layers[0])
                    if layer_class not in transformer_layer_classes:
                        transformer_layer_classes.append(layer_class)
                        main_logger_info(f"  Added {layer_class.__name__} to FSDP wrap policy")
        except Exception as e:
            logger.warning(f"[FSDP] Could not detect HFLM layer class: {e}")

        def lm_wrapper_wrap_policy(module):
            """FSDP wrap policy for LMModelWrapper."""
            # Wrap individual transformer layers in backbone and depformer
            return isinstance(module, tuple(transformer_layer_classes))

        auto_wrap_policy = functools.partial(
            torch_wrap.transformer_auto_wrap_policy,
            transformer_layer_cls=tuple(transformer_layer_classes),
        )

        wrapped_model = FullyShardedDataParallel(
            lm_wrapper,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            use_orig_params=True,
        )

        main_logger_info("  FSDP wrapping complete!")

    elif args.distributed_backend == "ddp":
        main_logger_info(f"  Wrapping with DDP over {get_world_size()} GPUs...")

        wrapped_model = DistributedDataParallel(
            lm_wrapper,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=False,
            broadcast_buffers=True,
        )

        main_logger_info("  DDP wrapping complete!")

    else:
        raise ValueError(f"Unknown distributed backend: {args.distributed_backend}")

    main_logger_info("=" * 70)
    log_train_params(wrapped_model)

    return wrapped_model


def _initialize_extended_modules(model: LMModel, param_dtype: torch.dtype) -> None:
    """
    Initialize extended modules by copying weights from original modules.

    When dynamically extending dep_q=8 → dep_q=16, the checkpoint only has
    weights for indices 0-7 (or 0-6 for depformer_emb). This function copies
    those weights to extended indices so they are properly initialized.

    Extended modules structure (from extend_moshi_modules_for_user_stream):
    - depformer_in: extend with copies of [0:8] → [8:16] = copies of [0:8]
    - depformer_emb: append(copy of [0]) then extend(copies of [0:7])
        → [7] = copy of [0], [8:15] = copies of [0:7]
    - linears: extend with copies of [0:8] → [8:16] = copies of [0:8]
    - For each depformer layer:
        - self_attn.in_projs: extend → [8:16] = copies of [0:8]
        - self_attn.out_projs: extend → [8:16] = copies of [0:8]
        - gating: extend → [8:16] = copies of [0:8]

    Args:
        model: LMModel with extended modules on meta device
        param_dtype: Target dtype for parameters
    """
    initialized_count = 0

    def copy_module_weights(src_module: torch.nn.Module, dst_module: torch.nn.Module, name: str):
        """Copy weights from source module to destination module."""
        nonlocal initialized_count

        src_params = list(src_module.named_parameters(recurse=True))
        dst_params = list(dst_module.named_parameters(recurse=True))

        for (src_name, src_param), (dst_name, dst_param) in zip(src_params, dst_params):
            if dst_param.is_meta:
                # Create new tensor with copied values
                new_param = torch.nn.Parameter(
                    src_param.clone().detach().to(dtype=param_dtype)
                )
                # Replace the meta parameter
                # Navigate to the correct sub-module to set the parameter
                parts = dst_name.split('.')
                target = dst_module
                for part in parts[:-1]:
                    target = getattr(target, part)
                target._parameters[parts[-1]] = new_param
                initialized_count += 1
                logger.debug(f"  Copied: {name}.{dst_name}")

    # 1. Initialize depformer_in[8:16] from depformer_in[0:8]
    if hasattr(model, 'depformer_in') and len(model.depformer_in) > 8:
        logger.info(f"Initializing depformer_in[8:16] (total: {len(model.depformer_in)})")
        for i in range(8, min(16, len(model.depformer_in))):
            src_idx = i - 8  # Map 8→0, 9→1, ..., 15→7
            copy_module_weights(
                model.depformer_in[src_idx],
                model.depformer_in[i],
                f"depformer_in[{i}]"
            )

    # 2. Initialize depformer_emb[7:15] from depformer_emb[0:7]
    # Extension does: append(copy of [0]) then extend(copies of [0:7])
    # So: [7] = copy of [0], [8]=copy of [0], [9]=copy of [1], ..., [14]=copy of [6]
    if hasattr(model, 'depformer_emb') and len(model.depformer_emb) > 7:
        logger.info(f"Initializing depformer_emb[7:15] (total: {len(model.depformer_emb)})")
        for i in range(7, min(15, len(model.depformer_emb))):
            if i == 7:
                src_idx = 0  # Index 7 is append(copy of [0])
            else:
                src_idx = i - 8  # Indices 8-14 are extend(copies of [0:7]): 8→0, 9→1, ..., 14→6
            copy_module_weights(
                model.depformer_emb[src_idx],
                model.depformer_emb[i],
                f"depformer_emb[{i}]"
            )

    # 3. Initialize linears[8:16] from linears[0:8]
    if hasattr(model, 'linears') and len(model.linears) > 8:
        logger.info(f"Initializing linears[8:16] (total: {len(model.linears)})")
        for i in range(8, min(16, len(model.linears))):
            src_idx = i - 8
            copy_module_weights(
                model.linears[src_idx],
                model.linears[i],
                f"linears[{i}]"
            )

    # 4. Initialize depformer layer-specific modules
    # Note: in_projs and out_projs are under layer.self_attn, not layer directly
    if hasattr(model, 'depformer'):
        for layer_idx, layer in enumerate(model.depformer.layers):
            # self_attn.in_projs[8:16]
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'in_projs'):
                if len(layer.self_attn.in_projs) > 8:
                    for i in range(8, min(16, len(layer.self_attn.in_projs))):
                        src_idx = i - 8
                        copy_module_weights(
                            layer.self_attn.in_projs[src_idx],
                            layer.self_attn.in_projs[i],
                            f"depformer.layers[{layer_idx}].self_attn.in_projs[{i}]"
                        )

            # self_attn.out_projs[8:16]
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_projs'):
                if len(layer.self_attn.out_projs) > 8:
                    for i in range(8, min(16, len(layer.self_attn.out_projs))):
                        src_idx = i - 8
                        copy_module_weights(
                            layer.self_attn.out_projs[src_idx],
                            layer.self_attn.out_projs[i],
                            f"depformer.layers[{layer_idx}].self_attn.out_projs[{i}]"
                        )

            # gating[8:16]
            if hasattr(layer, 'gating') and len(layer.gating) > 8:
                for i in range(8, min(16, len(layer.gating))):
                    src_idx = i - 8
                    copy_module_weights(
                        layer.gating[src_idx],
                        layer.gating[i],
                        f"depformer.layers[{layer_idx}].gating[{i}]"
                    )

    logger.info(f"Initialized {initialized_count} extended parameters")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for
      every trainable and non-trainable parameter group since this is a
      requirement for mixed requires_grad=True/False training. See:
      https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel, DistributedDataParallel]):
    """Log the number of trainable parameters in the model."""
    world_size = get_world_size()

    # For FSDP, parameters are sharded, so multiply by world_size
    # For DDP, parameters are replicated, so don't multiply
    if isinstance(model, FullyShardedDataParallel):
        num_params = world_size * sum(p.numel() for p in model.parameters())
        num_train_params = world_size * sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:
        # DDP or plain model
        inner_model = model.module if isinstance(model, DistributedDataParallel) else model
        num_params = sum(p.numel() for p in inner_model.parameters())
        num_train_params = sum(
            p.numel() for p in inner_model.parameters() if p.requires_grad
        )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo:
    https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError("Only Lora layers should be randomly initialized.")


def _load_and_prepare_model(
    args: TrainArgs,
    checkpointer_info: CheckpointInfo,
    param_dtype: torch.dtype,
) -> LMModel:
    """
    Load model weights and prepare for training.

    This is a shared helper for both FSDP and DDP model loading.
    For FSDP, only rank 0 loads weights; for DDP, all ranks load.

    Args:
        args: Training arguments
        checkpointer_info: Checkpoint information for loading weights
        param_dtype: Data type for model parameters

    Returns:
        Initialized LMModel (not wrapped)

    Note:
        Moshiko weights are trained with n_q=16 (17 codebooks), dep_q=8.
        We MUST keep n_q=16 to match the weight architecture.
        The data (9 codebooks) will be padded to 17 in train.py.

        Architecture:
        - n_q=16: 16 audio codebook positions (full-duplex mode)
        - dep_q=8: depformer only handles first 8 audio codebooks
        - For mono finetuning: only codebooks 0-8 are used (text + 8 audio)
        - Codebooks 9-16 are padded with zero_token_id (-1)
    """
    # Keep original n_q=16 to match moshiko weight architecture!
    # DO NOT change n_q or delays - weight loading requires exact match.
    # The dep_q=8 is already the default in _lm_kwargs.
    finetuning_overrides = {
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora": args.lora.enable,
        "lora_rank": args.lora.rank,
        "lora_scaling": args.lora.scaling,
        # CRITICAL: Do NOT override n_q, dep_q, or delays!
        # Moshiko weights expect n_q=16, dep_q=8, delays=[17 elements]
        # Data padding from 9 to 17 codebooks is done in train.py
    }

    main_logger_info("Model config: Using original moshiko architecture (n_q=16, dep_q=8)")
    main_logger_info("Data will be padded from 9 to 17 codebooks in training loop")

    with torch.device("meta"):
        model = checkpointer_info.get_moshi(
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides=finetuning_overrides,
            load_weight=False,
        )

    main_logger_info(f"Model initialized: n_q={model.n_q}, dep_q={model.dep_q}, num_codebooks={model.num_codebooks}")

    return model


def _load_user_stream_model(
    args: TrainArgs,
    param_dtype: torch.dtype,
    checkpointer_info: Optional[CheckpointInfo] = None,
) -> tuple[LMModel, dict, str, bool]:
    """
    Load or dynamically create a model with user stream extension (dep_q=16).

    Supports two modes:
    1. Pre-initialized mode: Load from korean.initialized_model_path
       (created by tools/init_korean_moshi.py)
    2. Dynamic extension mode: Load original dep_q=8 model and extend in memory
       (no pre-initialization required)

    Args:
        args: Training arguments
        param_dtype: Data type for model parameters
        checkpointer_info: Optional checkpoint info for dynamic extension

    Returns:
        Tuple of (LMModel, lm_kwargs_dict, model_file_path, needs_dynamic_extension)
        - needs_dynamic_extension: True if model needs to be extended after weight loading

    Note:
        For dynamic extension, the model is created with dep_q=8 structure,
        weights are loaded, then extend_moshi_modules_for_user_stream() is called.
    """
    model_path = getattr(args.korean, 'initialized_model_path', None)

    # =========================================================================
    # Mode 1: Pre-initialized model exists
    # =========================================================================
    if model_path:
        model_dir = Path(model_path)

        # Support both directory and file path
        if model_dir.is_file():
            model_file = model_dir
            config_file = model_dir.parent / "config.json"
        else:
            model_file = model_dir / "model.safetensors"
            config_file = model_dir / "config.json"

        if model_file.exists():
            main_logger_info(f"Using pre-initialized user stream model: {model_file}")

            # Load config if exists
            lm_kwargs = deepcopy(_lm_kwargs)
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    if "lm_kwargs" in config:
                        lm_kwargs.update(config["lm_kwargs"])
                main_logger_info(f"Loaded config from: {config_file}")
            else:
                lm_kwargs.update({"dep_q": 16, "depformer_context": 16})
                main_logger_info("No config.json found, using default dep_q=16")

            # Apply training-specific overrides
            # IMPORTANT: LoRA parameters should NOT be passed to LMModel.__init__
            # The moshi library handles LoRA separately via replace_all_linear_with_lora()
            # after model creation. See moshi/models/loaders.py:get_moshi_lm()
            lm_kwargs.update({
                "gradient_checkpointing": args.gradient_checkpointing,
                # NOTE: lora, lora_rank, lora_scaling are handled separately after model creation
            })

            # Create model on meta device with dep_q=16
            with torch.device("meta"):
                model = LMModel(**lm_kwargs)

            main_logger_info(
                f"Pre-initialized model loaded: dep_q={model.dep_q}, "
                f"num_codebooks={model.num_codebooks}, linears={len(model.linears)}"
            )

            # CRITICAL: Validate that pre-initialized model is properly extended
            if model.dep_q != 16 or len(model.linears) != 16:
                main_logger_info("=" * 60)
                main_logger_info("[WARNING] Pre-initialized model has incorrect architecture!")
                main_logger_info(f"  Expected: dep_q=16, linears=16 (user stream mode)")
                main_logger_info(f"  Actual:   dep_q={model.dep_q}, linears={len(model.linears)}")
                main_logger_info("  Falling back to dynamic extension mode...")
                main_logger_info("=" * 60)
                # Fall through to dynamic extension
            else:
                return model, lm_kwargs, str(model_file), False  # No dynamic extension needed
        else:
            main_logger_info(
                f"Warning: initialized_model_path set but file not found: {model_file}"
            )
            main_logger_info("Falling back to dynamic extension mode...")

    # =========================================================================
    # Mode 2: Dynamic extension (no pre-initialization required)
    # =========================================================================
    if not MODEL_UTILS_AVAILABLE:
        raise ImportError(
            "Dynamic model extension requires tools.model_utils module. "
            "Either:\n"
            "  1. Run: python tools/init_korean_moshi.py --save_dir ./models/k-moshi-init --extend_modules_for_user_stream\n"
            "  2. Or ensure tools/model_utils.py is in the Python path"
        )

    if checkpointer_info is None:
        raise ValueError(
            "Dynamic extension requires checkpointer_info to locate original model weights. "
            "Please provide checkpointer_info or use a pre-initialized model."
        )

    main_logger_info("=" * 60)
    main_logger_info("DYNAMIC MODEL EXTENSION MODE")
    main_logger_info("=" * 60)
    main_logger_info("No pre-initialized model found. Will dynamically extend dep_q=8 → dep_q=16")

    # Get original model weights path
    original_model_file = checkpointer_info.moshi_weights
    main_logger_info(f"Original model weights: {original_model_file}")

    # Create original dep_q=8 model with standard kwargs
    # IMPORTANT: LoRA parameters should NOT be passed to LMModel.__init__
    # The moshi library handles LoRA separately via replace_all_linear_with_lora()
    # after model creation. See moshi/models/loaders.py:get_moshi_lm()
    lm_kwargs = deepcopy(_lm_kwargs)
    lm_kwargs.update({
        "gradient_checkpointing": args.gradient_checkpointing,
        # NOTE: lora, lora_rank, lora_scaling are handled separately after model creation
    })

    # Create model on meta device with original dep_q=8
    with torch.device("meta"):
        model = LMModel(**lm_kwargs)

    main_logger_info(
        f"Original model created: dep_q={model.dep_q}, "
        f"num_codebooks={model.num_codebooks}"
    )
    main_logger_info("Model will be extended to dep_q=16 after weight loading")

    # Return with flag indicating dynamic extension is needed
    return model, lm_kwargs, str(original_model_file), True  # Needs dynamic extension


def _set_requires_grad(model: torch.nn.Module, args: TrainArgs):
    """Set requires_grad based on LoRA or full finetuning configuration."""
    if args.lora.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def get_fsdp_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> FullyShardedDataParallel | LMModel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) LMModel or a non sharded LMModel if one GPU available.

    Args:
        args (TrainArgs): A configuration object containing training arguments
            and settings. Key attributes include:
            - param_dtype: The data type for model parameters (e.g., "bfloat16", "float32").
            - gradient_checkpointing: Whether to enable gradient checkpointing.
            - lora: Configuration for LoRA fine-tuning, including enabling, rank, and scaling.
            - full_finetuning: Whether to enable full model fine-tuning or only LoRA fine-tuning.
            - korean: Korean finetuning configuration (user stream, tokenizer, etc.)
        checkpointer_info: provide the initial checkpoint to train from.

    Notes:
        - The function uses meta-device initialization for memory efficiency.
        - Then parameters are initialized on the first GPU (rank=0) only.
        - For user stream mode (korean.enable_user_stream=True), loads from
          korean.initialized_model_path instead of the default Moshiko weights.
    """

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    # Check if user stream mode is enabled - with explicit logging
    enable_user_stream = getattr(args.korean, 'enable_user_stream', False)
    initialized_model_path = getattr(args.korean, 'initialized_model_path', None)

    needs_dynamic_extension = False

    # DEBUG: Log Korean config for verification
    main_logger_info("=" * 60)
    main_logger_info("[FSDP CONFIG CHECK]")
    main_logger_info(f"  enable_user_stream: {enable_user_stream}")
    main_logger_info(f"  initialized_model_path: {initialized_model_path}")
    main_logger_info(f"  MODEL_UTILS_AVAILABLE: {MODEL_UTILS_AVAILABLE}")
    main_logger_info("=" * 60)

    if enable_user_stream:
        main_logger_info("User stream mode enabled - loading extended model (dep_q=16)")
        model, lm_kwargs, model_file, needs_dynamic_extension = _load_user_stream_model(
            args, param_dtype, checkpointer_info
        )
        moshi_weight = str(model_file)
    else:
        model = _load_and_prepare_model(args, checkpointer_info, param_dtype)
        moshi_weight = checkpointer_info.moshi_weights

    # ==========================================================================
    # CRITICAL FIX: Dynamic extension must happen on ALL ranks BEFORE weight loading
    # This ensures all ranks have the same model architecture for FSDP sync.
    #
    # The model is on meta device, so extension only changes the architecture
    # (adds new module placeholders), not actual weights.
    # ==========================================================================
    if needs_dynamic_extension:
        if get_rank() == 0:
            logger.info("=" * 60)
            logger.info("APPLYING DYNAMIC MODEL EXTENSION (ALL RANKS)")
            logger.info("=" * 60)
            logger.info(f"Before extension: dep_q={model.dep_q}, depformer_in={len(model.depformer_in)}")

        # ALL ranks extend model architecture (meta device, no actual weights)
        model = extend_moshi_modules_for_user_stream(model)

        if get_rank() == 0:
            logger.info(f"After extension: dep_q={model.dep_q}, depformer_in={len(model.depformer_in)}, depformer_norms={len(model.depformer_norms)}, linears={len(model.linears)}")
            logger.info("Dynamic extension completed on all ranks!")
            logger.info("=" * 60)

    # Now all ranks have identical model architecture (dep_q=16 if extended)
    # RANK 0 loads weights, other ranks wait with meta device model

    if get_rank() == 0:
        assert _is_safetensors(moshi_weight), f"Model is not safetensors: {moshi_weight}"
        model_state_dict = safetensors.torch.load_file(moshi_weight)

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            model_state_dict[k] = v.to(param_dtype)

        model.load_state_dict(model_state_dict, strict=False, assign=True)

        # =======================================================================
        # CRITICAL: Initialize extended modules that are still on meta device
        # The checkpoint only has dep_q=8 modules, so extended modules (indices 8-15)
        # are not loaded and remain as meta tensors. We need to initialize them
        # by copying weights from the original modules (indices 0-7).
        # =======================================================================
        if needs_dynamic_extension:
            logger.info("Initializing extended modules (copying from original modules)...")
            _initialize_extended_modules(model, param_dtype)
            logger.info("Extended modules initialized!")

        # CRITICAL: Validate model architecture for user stream mode
        if enable_user_stream:
            expected_dep_q = 16
            expected_linears = 16
            expected_norms = 16
            actual_dep_q = getattr(model, 'dep_q', None)
            actual_linears = len(model.linears) if hasattr(model, 'linears') else 0
            actual_norms = len(model.depformer_norms) if hasattr(model, 'depformer_norms') else 0

            if actual_dep_q != expected_dep_q or actual_linears != expected_linears or actual_norms != expected_norms:
                logger.error("=" * 60)
                logger.error("[ARCHITECTURE MISMATCH DETECTED]")
                logger.error(f"  Expected: dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms}")
                logger.error(f"  Actual:   dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms}")
                logger.error("=" * 60)
                raise RuntimeError(
                    f"Model architecture mismatch for user stream mode!\n"
                    f"  Expected: dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms}\n"
                    f"  Actual:   dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms}\n\n"
                    f"Solutions:\n"
                    f"  1. Re-create the extended model:\n"
                    f"     python -m tools.init_korean_moshi --save_dir ./models/k-moshi-init --extend_modules_for_user_stream\n"
                    f"  2. Or set initialized_model_path: null to use dynamic extension"
                )
            else:
                logger.info(f"[ARCHITECTURE OK] dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms} (user stream mode)")

        if args.lora.enable and not args.full_finetuning:
            logger.info("Initializing lora layers ...")
            # initialize LoRA layers
            initialize_lora_parameters(model, param_dtype)

        # Check for remaining meta parameters
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            logger.error(f"CRITICAL: {len(meta_params)} parameters still on meta device!")
            logger.error(f"First 10: {meta_params[:10]}")
            raise RuntimeError(
                f"Failed to initialize all parameters. {len(meta_params)} params still on meta: {meta_params[:5]}..."
            )

        assert all(p.dtype == param_dtype for p in model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in model.parameters()), (
            "All parameters should be on meta"
        )

    torch.distributed.barrier()

    # only finetune LoRA parameters and freeze before wrapping
    _set_requires_grad(model, args)

    if get_world_size() == 1:
        return model.cuda()

    auto_wrap_policy = get_fsdp_policy(args.lora.enable)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs using FSDP...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
    )

    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)

    return wrapped_model


def get_ddp_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> DistributedDataParallel | LMModel:
    """
    Initializes and returns a DistributedDataParallel (DDP) wrapped LMModel.

    DDP replicates the full model on each GPU and synchronizes gradients.
    This is simpler than FSDP but requires more memory per GPU.

    For LoRA finetuning on A100 80GB, DDP works well and is more compatible
    with MPI launchers.

    Args:
        args (TrainArgs): Training configuration
        checkpointer_info: Checkpoint information for loading weights

    Returns:
        DDP-wrapped model or plain model for single GPU

    Notes:
        For user stream mode (korean.enable_user_stream=True), loads from
        korean.initialized_model_path instead of the default Moshiko weights.
    """

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    main_logger_info("Loading model for DDP...")

    # Check if user stream mode is enabled - with explicit logging
    enable_user_stream = getattr(args.korean, 'enable_user_stream', False)
    initialized_model_path = getattr(args.korean, 'initialized_model_path', None)
    needs_dynamic_extension = False

    # DEBUG: Log Korean config for verification
    main_logger_info("=" * 60)
    main_logger_info("[DDP CONFIG CHECK]")
    main_logger_info(f"  enable_user_stream: {enable_user_stream}")
    main_logger_info(f"  initialized_model_path: {initialized_model_path}")
    main_logger_info(f"  MODEL_UTILS_AVAILABLE: {MODEL_UTILS_AVAILABLE}")
    main_logger_info("=" * 60)

    if enable_user_stream:
        main_logger_info("User stream mode enabled - loading extended model (dep_q=16)")
        model, lm_kwargs, model_file, needs_dynamic_extension = _load_user_stream_model(
            args, param_dtype, checkpointer_info
        )
        moshi_weight = str(model_file)
        main_logger_info(f"Model initialized: n_q={model.n_q}, dep_q={model.dep_q}, num_codebooks={model.num_codebooks}")
    else:
        # Keep original n_q=16 to match moshiko weight architecture!
        # DO NOT change n_q or delays - weight loading requires exact match.
        finetuning_overrides = {
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora": args.lora.enable,
            "lora_rank": args.lora.rank,
            "lora_scaling": args.lora.scaling,
            # CRITICAL: Do NOT override n_q, dep_q, or delays!
            # Moshiko weights expect n_q=16, dep_q=8, delays=[17 elements]
            # Data padding from 9 to 17 codebooks is done in train.py
        }

        main_logger_info("Model config: Using original moshiko architecture (n_q=16, dep_q=8)")
        main_logger_info("Data will be padded from 9 to 17 codebooks in training loop")

        # Load model on meta device first
        with torch.device("meta"):
            model = checkpointer_info.get_moshi(
                device="meta",
                dtype=param_dtype,
                lm_kwargs_overrides=finetuning_overrides,
                load_weight=False,
            )

        main_logger_info(f"Model initialized: n_q={model.n_q}, dep_q={model.dep_q}, num_codebooks={model.num_codebooks}")

        # For DDP, all ranks load the full model
        moshi_weight = checkpointer_info.moshi_weights

    assert _is_safetensors(moshi_weight), f"Model is not safetensors: {moshi_weight}"

    main_logger_info(f"Loading weights from {moshi_weight}...")
    model_state_dict = safetensors.torch.load_file(moshi_weight)

    # Convert to target dtype
    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(param_dtype)

    # NOTE: Do NOT call model.to_empty() here!
    # to_empty() moves ALL parameters to CPU with uninitialized (garbage) values.
    # This breaks LoRA initialization because:
    #   1. to_empty() moves LoRA params to CPU with garbage values
    #   2. load_state_dict() doesn't have LoRA weights (they're not in checkpoint)
    #   3. initialize_lora_parameters() checks is_meta but LoRA params are on CPU now
    #   4. Result: LoRA params stay uninitialized with NaN/garbage values!
    #
    # Instead, keep model on meta device and use assign=True which will:
    #   1. Replace matched params (base model weights) with checkpoint values
    #   2. Keep unmatched params (LoRA layers) on meta device
    #   3. initialize_lora_parameters() will then properly initialize LoRA layers

    # Load weights directly from meta device with assign=True
    # assign=True replaces parameters in-place, materializing them from meta
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False, assign=True)

    # Dynamic extension: extend model from dep_q=8 to dep_q=16 after weight loading
    if needs_dynamic_extension:
        main_logger_info("=" * 60)
        main_logger_info("APPLYING DYNAMIC MODEL EXTENSION (DDP)")
        main_logger_info("=" * 60)
        main_logger_info(f"Before extension: dep_q={model.dep_q}, depformer_in={len(model.depformer_in)}, linears={len(model.linears)}")

        # Extend model modules for user stream
        # This uses deepcopy of already-loaded modules, so extended modules will have weights
        model = extend_moshi_modules_for_user_stream(model)

        main_logger_info(f"After extension: dep_q={model.dep_q}, depformer_in={len(model.depformer_in)}, depformer_norms={len(model.depformer_norms)}, linears={len(model.linears)}")

        # Verify extension was successful
        if len(model.linears) != 16:
            raise RuntimeError(
                f"Dynamic model extension failed! Expected 16 linears, got {len(model.linears)}. "
                f"Check tools/model_utils.py:extend_moshi_modules_for_user_stream()"
            )

        main_logger_info("Dynamic extension completed successfully!")
        main_logger_info("=" * 60)

    # CRITICAL: Validate model architecture for user stream mode
    if enable_user_stream:
        expected_dep_q = 16
        expected_linears = 16
        expected_norms = 16
        actual_dep_q = getattr(model, 'dep_q', None)
        actual_linears = len(model.linears) if hasattr(model, 'linears') else 0
        actual_norms = len(model.depformer_norms) if hasattr(model, 'depformer_norms') else 0

        if actual_dep_q != expected_dep_q or actual_linears != expected_linears or actual_norms != expected_norms:
            main_logger_info("=" * 60)
            main_logger_info("[ARCHITECTURE MISMATCH DETECTED]")
            main_logger_info(f"  Expected: dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms}")
            main_logger_info(f"  Actual:   dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms}")
            main_logger_info("=" * 60)
            raise RuntimeError(
                f"Model architecture mismatch for user stream mode!\n"
                f"  Expected: dep_q={expected_dep_q}, linears={expected_linears}, depformer_norms={expected_norms}\n"
                f"  Actual:   dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms}\n\n"
                f"Possible causes:\n"
                f"  1. Pre-initialized model at '{initialized_model_path}' is not properly extended\n"
                f"  2. Dynamic extension failed\n"
                f"  3. Config mismatch: enable_user_stream=True but model has dep_q=8\n\n"
                f"Solutions:\n"
                f"  1. Re-create the extended model:\n"
                f"     python -m tools.init_korean_moshi --save_dir ./models/k-moshi-init --extend_modules_for_user_stream\n"
                f"  2. Or set initialized_model_path: null to use dynamic extension\n"
                f"  3. Or set enable_user_stream: false for mono training"
            )
        else:
            main_logger_info(f"[ARCHITECTURE OK] dep_q={actual_dep_q}, linears={actual_linears}, depformer_norms={actual_norms} (user stream mode)")

    # Log missing and unexpected keys for debugging
    if missing_keys:
        lora_missing = [k for k in missing_keys if "lora" in k.lower()]
        non_lora_missing = [k for k in missing_keys if "lora" not in k.lower()]

        main_logger_info(f"[WEIGHT LOADING] Missing keys: {len(lora_missing)} LoRA + {len(non_lora_missing)} non-LoRA")

        if non_lora_missing:
            logger.warning(
                f"[WEIGHT LOADING] Non-LoRA missing keys ({len(non_lora_missing)}): "
                f"{non_lora_missing[:5]}... - These parameters may be UNINITIALIZED!"
            )
    if unexpected_keys:
        main_logger_info(f"[WEIGHT LOADING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    # After load_state_dict with assign=True:
    # - Base model params: loaded from checkpoint, on CPU
    # - LoRA params: still on meta device (expected, will be initialized next)
    meta_params = [(n, p) for n, p in model.named_parameters() if p.is_meta]
    lora_meta = [n for n, _ in meta_params if "lora" in n.lower()]
    non_lora_meta = [n for n, _ in meta_params if "lora" not in n.lower()]

    if non_lora_meta:
        raise RuntimeError(
            f"CRITICAL: {len(non_lora_meta)} non-LoRA parameters still on meta device! "
            f"First 5: {non_lora_meta[:5]}"
        )

    main_logger_info(f"[WEIGHT LOADING] Base weights loaded. {len(lora_meta)} LoRA params on meta (will be initialized)")

    # DIAGNOSTIC: Check for NaN/Inf in loaded base weights (skip meta LoRA params)
    nan_params = []
    inf_params = []
    for name, param in model.named_parameters():
        if param.is_meta:  # Skip meta tensors (LoRA params not yet initialized)
            continue
        if param.numel() > 0:
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

    if nan_params:
        logger.error(f"[WEIGHT LOADING] NaN found in {len(nan_params)} base weight params: {nan_params[:5]}...")
    if inf_params:
        logger.error(f"[WEIGHT LOADING] Inf found in {len(inf_params)} base weight params: {inf_params[:5]}...")
    if not nan_params and not inf_params:
        main_logger_info("[WEIGHT LOADING] All base weights are valid (no NaN/Inf)")

    if args.lora.enable and not args.full_finetuning:
        main_logger_info("Initializing LoRA layers...")
        initialize_lora_parameters(model, param_dtype)

        # Verify LoRA initialization succeeded
        meta_after = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_after:
            raise RuntimeError(
                f"CRITICAL: {len(meta_after)} params still on meta after LoRA init! "
                f"First 5: {meta_after[:5]}"
            )

        # Check for NaN/Inf in LoRA parameters
        lora_nan = []
        lora_inf = []
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.numel() > 0:
                if torch.isnan(param).any():
                    lora_nan.append(name)
                if torch.isinf(param).any():
                    lora_inf.append(name)

        if lora_nan:
            raise RuntimeError(
                f"CRITICAL: NaN in {len(lora_nan)} LoRA params after init! "
                f"First 5: {lora_nan[:5]}"
            )
        if lora_inf:
            raise RuntimeError(
                f"CRITICAL: Inf in {len(lora_inf)} LoRA params after init! "
                f"First 5: {lora_inf[:5]}"
            )

        main_logger_info("[LORA INIT] All LoRA parameters initialized successfully (no NaN/Inf)")

    # Set requires_grad before moving to GPU
    _set_requires_grad(model, args)

    # Move to GPU
    device = torch.cuda.current_device()
    model = model.to(device)

    main_logger_info("Model loaded to GPU!")

    if get_world_size() == 1:
        log_train_params(model)
        return model

    # Wrap with DDP
    main_logger_info(f"Wrapping model with DDP over {get_world_size()} GPUs...")

    # Find parameters that require gradients for DDP
    # DDP needs to know which parameters to sync
    wrapped_model = DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False,  # Set to True if some params are unused
        broadcast_buffers=True,
    )

    main_logger_info("Model wrapped with DDP!")
    log_train_params(wrapped_model)

    return wrapped_model


def get_distributed_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> Union[FullyShardedDataParallel, DistributedDataParallel, LMModel]:
    """
    Get the appropriate distributed model based on configuration.

    This is the main entry point for getting a distributed model.
    It selects between:
    1. Backbone type: "moshi" (legacy LMModel) vs "hf_lm" (modular backbone)
    2. Distributed backend: FSDP vs DDP

    Args:
        args: Training arguments with distributed_backend and backbone settings
        checkpointer_info: Checkpoint information for loading weights

    Returns:
        Wrapped model (FSDP, DDP, or plain model for single GPU)

    Raises:
        NotImplementedError: For non-moshi backbones (Phase 3)
        ValueError: For unknown distributed backend
    """
    # ==========================================================================
    # Phase 2: Backbone Type Routing
    # ==========================================================================
    # Check if modular backbone system should be used
    use_modular = _should_use_modular_backbone(args)

    if use_modular:
        # Modular backbone system (HFLM, custom backbones)
        main_logger_info(f"[BACKBONE] Modular backbone requested: {args.backbone.type}")

        # Validate configuration
        validation_messages = BackboneFactory.validate_config(args.backbone)
        for msg in validation_messages:
            if msg.startswith("ERROR"):
                raise ValueError(msg)
            else:
                logger.warning(msg)

        # Phase 3: LMModelWrapper implementation
        return _get_modular_backbone_model(args, checkpointer_info)

    # ==========================================================================
    # Legacy Moshi Path (backward compatible)
    # ==========================================================================
    main_logger_info(f"[BACKBONE] Using legacy LMModel path (type={args.backbone.type})")

    if args.distributed_backend == "fsdp":
        return get_fsdp_model(args, checkpointer_info)
    elif args.distributed_backend == "ddp":
        return get_ddp_model(args, checkpointer_info)
    else:
        raise ValueError(f"Unknown distributed backend: {args.distributed_backend}")


def get_unwrapped_model(
    model: Union[FullyShardedDataParallel, DistributedDataParallel, LMModel]
) -> LMModel:
    """
    Get the underlying LMModel from a wrapped distributed model.

    Args:
        model: Wrapped or plain model

    Returns:
        The underlying LMModel
    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    elif isinstance(model, FullyShardedDataParallel):
        # For FSDP, the module is directly accessible but parameters are sharded
        return model
    else:
        return model
