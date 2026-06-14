"""
K-Moshi Model Utilities

Core utilities for Korean Moshi full finetuning:
- User stream extension (dep_q=8 → 16) for full-duplex training
- User stream removal for serving (dep_q=16 → 8)
- Text embedding reinitialization for Korean tokenizer

Architecture Overview:
    Original Moshiko: n_q=16, dep_q=8 (only Moshi's 8 audio codebooks in depformer)
    Extended K-Moshi: n_q=16, dep_q=16 (Moshi's 8 + User's 8 audio codebooks)

    The depformer modules need to be extended to model user stream:
    - depformer_in: 8 → 16 embeddings
    - depformer_emb: 7 → 15 embeddings (semantic tokens predicted from text, not depformer_emb)
    - depformer_norms: 8 → 16 LayerNorm modules (CRITICAL: used in forward_depformer_training)
    - depformer.layers:
        - self_attn.in_projs: ModuleList of 8 → 16 Linear layers (Q,K,V projections per position)
        - self_attn.out_projs: ModuleList of 8 → 16 Linear layers (output projections per position)
        - gating: 8 → 16 gating modules
    - linears: 8 → 16 output projections

Note on StreamingMultiheadAttention:
    Moshi uses `depformer_weights_per_step=True` which means the attention uses
    ModuleLists (in_projs, out_projs) with separate Linear layers for each codebook
    position, NOT single weight tensors. This is critical for the extension logic.
"""

from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from moshi.models import LMModel


def extend_moshi_modules_for_user_stream(lm: LMModel, inplace: bool = True) -> LMModel:
    """
    Extend the depth transformer's modules to model user stream.

    This enables full-duplex training where both Moshi's and User's audio
    streams are modeled simultaneously.

    Changes made (dep_q=8 → dep_q=16):
        1. depformer_in: 8 → 16 (duplicate all input embeddings)
        2. depformer_emb: 7 → 15 (7 original + 1 for user semantic + 7 for user acoustic)
        3. depformer_norms: 8 → 16 (duplicate LayerNorm modules - CRITICAL!)
        4. depformer.layers:
            - self_attn.in_projs: ModuleList 8 → 16 (duplicate Linear layers)
            - self_attn.out_projs: ModuleList 8 → 16 (duplicate Linear layers)
            - gating: 8 → 16 gating modules
        5. linears: 8 → 16 output projection layers

    Args:
        lm: Original LMModel with dep_q=8
        inplace: If True, modify lm in-place (saves memory for large models).
                 If False, create a deep copy first. Default: True

    Returns:
        Extended LMModel with dep_q=16 capability

    Note:
        The depformer_emb doesn't have an embedding to encode Moshi's last acoustic
        token (for predicting user's semantic token) because Moshi's semantic token
        is predicted from text token. We reuse the first embedding in depformer_emb
        (which is for predicting the first acoustic token) for this purpose.

        StreamingMultiheadAttention uses in_projs and out_projs ModuleLists
        (not single in_proj_weight/out_proj tensors) when depformer_weights_per_step=True.
    """
    if inplace:
        lm_us = lm
    else:
        lm_us = deepcopy(lm)

    # For in-place modification, we need to save copies of original modules
    # BEFORE modifying the lists, otherwise we'd copy already-modified lists
    original_depformer_in = [deepcopy(m) for m in lm.depformer_in]
    original_depformer_emb = [deepcopy(m) for m in lm.depformer_emb]
    original_depformer_norms = [deepcopy(m) for m in lm.depformer_norms]  # CRITICAL: for forward_depformer_training
    original_linears = [deepcopy(m) for m in lm.linears]
    original_gatings = [[deepcopy(g) for g in layer.gating] for layer in lm.depformer.layers]

    # Save original attention modules for each layer
    # in_projs and out_projs are ModuleLists with 8 Linear layers each
    original_in_projs = [[deepcopy(proj) for proj in layer.self_attn.in_projs] for layer in lm.depformer.layers]
    original_out_projs = [[deepcopy(proj) for proj in layer.self_attn.out_projs] for layer in lm.depformer.layers]

    # 1. Extend depformer_in (input embeddings for each codebook)
    # Original: 8 embeddings for Moshi's 8 audio codebooks
    # Extended: 8 (Moshi) + 8 (User) = 16 embeddings
    lm_us.depformer_in.extend(original_depformer_in)

    # 2. Extend depformer_emb (embeddings for depformer's sequential prediction)
    # Original: 7 embeddings (indices 0-6 for predicting codebooks 1-7, semantic from text)
    # Extended: 7 (Moshi) + 1 (User semantic) + 7 (User acoustic) = 15 embeddings
    #
    # The first user embedding (index 7) is for predicting user's first acoustic token,
    # which comes right after Moshi's last acoustic token. We use a copy of
    # depformer_emb[0] since there's no explicit embedding for this transition.
    lm_us.depformer_emb.append(deepcopy(original_depformer_emb[0]))  # For user semantic → first acoustic
    lm_us.depformer_emb.extend(original_depformer_emb)     # For user acoustic 1-7

    # 3. Extend depformer_norms (LayerNorm for each codebook output)
    # CRITICAL: forward_depformer_training accesses depformer_norms[cb_index]
    # Original: 8 LayerNorms (for Moshi's 8 audio codebooks)
    # Extended: 8 (Moshi) + 8 (User) = 16 LayerNorms
    lm_us.depformer_norms.extend(original_depformer_norms)

    # 4. Extend depformer layers (self-attention and gating)
    for i, layer in enumerate(lm_us.depformer.layers):
        # 4.1 Extend self_attn.in_projs (ModuleList of Linear layers)
        # Each Linear projects to Q, K, V for one codebook position
        # Original: 8 Linear layers → Extended: 16 Linear layers
        layer.self_attn.in_projs.extend(original_in_projs[i])

        # 4.2 Extend self_attn.out_projs (ModuleList of Linear layers)
        # Each Linear is the output projection for one codebook position
        # Original: 8 Linear layers → Extended: 16 Linear layers
        layer.self_attn.out_projs.extend(original_out_projs[i])

        # 4.3 Extend gating modules using pre-saved copies
        # Each position has its own gating: 8 → 16 modules
        layer.gating.extend(original_gatings[i])

    # 5. Extend linears (output projections to codebook vocabularies)
    # Original: 8 linears for Moshi's 8 audio codebooks
    # Extended: 8 (Moshi) + 8 (User) = 16 linears
    lm_us.linears.extend(original_linears)

    # 6. Update dep_q attribute to reflect extended model
    # This is critical for training loop and loss calculation
    if hasattr(lm_us, 'dep_q'):
        lm_us.dep_q = 16
    if hasattr(lm_us, 'depformer_context'):
        lm_us.depformer_context = 16

    return lm_us


def remove_moshi_modules_for_user_stream(lm_us: LMModel) -> LMModel:
    """
    Remove user stream modules from the extended depth transformer.

    This is the reverse operation of extend_moshi_modules_for_user_stream(),
    used when converting a trained model back to the original format for serving.

    Changes made (dep_q=16 → dep_q=8):
        1. depformer_in: 16 → 8 (keep first 8)
        2. depformer_emb: 15 → 7 (keep first 7)
        3. depformer_norms: 16 → 8 (keep first 8)
        4. depformer.layers:
            - self_attn.in_projs: ModuleList 16 → 8 (keep first 8)
            - self_attn.out_projs: ModuleList 16 → 8 (keep first 8)
            - gating: 16 → 8
        5. linears: 16 → 8 (keep first 8)

    Args:
        lm_us: Extended LMModel with dep_q=16

    Returns:
        Original LMModel architecture with dep_q=8
    """
    lm = deepcopy(lm_us)

    # 1. Reduce depformer_in (keep first 8)
    lm.depformer_in = nn.ModuleList(list(lm.depformer_in)[:8])

    # 2. Reduce depformer_emb (keep first 7)
    lm.depformer_emb = nn.ModuleList(list(lm.depformer_emb)[:7])

    # 3. Reduce depformer_norms (keep first 8)
    lm.depformer_norms = nn.ModuleList(list(lm.depformer_norms)[:8])

    # 4. Reduce depformer layers
    for layer in lm.depformer.layers:
        # 4.1 Reduce self_attn.in_projs (keep first 8 Linear layers)
        layer.self_attn.in_projs = nn.ModuleList(list(layer.self_attn.in_projs)[:8])

        # 4.2 Reduce self_attn.out_projs (keep first 8 Linear layers)
        layer.self_attn.out_projs = nn.ModuleList(list(layer.self_attn.out_projs)[:8])

        # 4.3 Reduce gating (keep first 8)
        layer.gating = nn.ModuleList(list(layer.gating)[:8])

    # 5. Reduce linears (keep first 8)
    lm.linears = nn.ModuleList(list(lm.linears)[:8])

    # 6. Update dep_q attribute
    if hasattr(lm, 'dep_q'):
        lm.dep_q = 8
    if hasattr(lm, 'depformer_context'):
        lm.depformer_context = 8

    return lm


def init_embedding_module(
    emb: nn.Embedding,
    retain_token_ids: List[int],
) -> nn.Embedding:
    """
    Reinitialize text embedding module with Gaussian distribution.

    When switching to a new tokenizer (e.g., Korean), the original English
    embeddings are no longer meaningful. This function:
    1. Computes mean and covariance of original embeddings
    2. Samples new embeddings from multivariate Gaussian
    3. Preserves specified special token embeddings (e.g., PAD, BOS, EOS)

    Args:
        emb: Original embedding module to reinitialize
        retain_token_ids: Token IDs whose embeddings should be preserved
                         Default: [0, 3, 32000] for special tokens

    Returns:
        Reinitialized embedding module

    Mathematical Details:
        - Mean vector μ: average of all original embeddings
        - Covariance Σ: (E - μ)ᵀ(E - μ) / vocab_size
        - New embeddings: sampled from N(μ, 1e-5 * Σ)
        - The 1e-5 scaling prevents extreme values while maintaining structure

    Note:
        MultivariateNormal only supports float32, so we temporarily convert
        and then cast back to the original dtype.
    """
    dtype = emb.weight.dtype
    emb_weights = emb.weight.data
    vocab_size = emb_weights.size(0)
    embed_dim = emb_weights.size(1)

    # Compute statistics in float32 for numerical stability
    weights_f32 = emb_weights.to(torch.float32)

    # Mean vector: [embed_dim]
    mean = weights_f32.mean(dim=0)

    # Covariance matrix: [embed_dim, embed_dim]
    # Σ = (X - μ)ᵀ(X - μ) / n
    centered = weights_f32 - mean.unsqueeze(0)
    sigma = (centered.T @ centered) / vocab_size

    # Create multivariate Gaussian distribution
    # Scale covariance by 1e-5 to prevent extreme values
    try:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mean,
            covariance_matrix=1e-5 * sigma,
        )
    except RuntimeError as e:
        # If covariance is not positive definite, add small diagonal
        print(f"Warning: Covariance matrix issue ({e}), adding regularization")
        sigma_reg = 1e-5 * sigma + 1e-6 * torch.eye(embed_dim, dtype=torch.float32, device=sigma.device)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mean,
            covariance_matrix=sigma_reg,
        )

    # Sample new embeddings
    new_weights = torch.stack(
        [dist.sample() for _ in range(vocab_size)],
        dim=0,
    ).to(dtype)

    # Preserve specified token embeddings (special tokens)
    for token_id in retain_token_ids:
        if token_id >= vocab_size:
            raise ValueError(
                f"Token ID {token_id} is out of range for vocab_size {vocab_size}"
            )
        new_weights[token_id] = emb_weights[token_id]

    # Update embedding weights
    emb.weight.data = new_weights

    return emb


def validate_extended_model(lm: LMModel) -> bool:
    """
    Validate that a model has been correctly extended for user stream.

    Checks:
        - depformer_in has 16 embeddings
        - depformer_emb has 15 embeddings
        - depformer_norms has 16 LayerNorm modules
        - All depformer layers have:
            - 16 in_projs Linear layers
            - 16 out_projs Linear layers
            - 16 gating modules
        - linears has 16 output projections

    Args:
        lm: LMModel to validate

    Returns:
        True if model is correctly extended, False otherwise

    Raises:
        ValueError: If validation fails with detailed error message
    """
    errors = []

    # Check depformer_in
    if len(lm.depformer_in) != 16:
        errors.append(f"depformer_in: expected 16, got {len(lm.depformer_in)}")

    # Check depformer_emb
    if len(lm.depformer_emb) != 15:
        errors.append(f"depformer_emb: expected 15, got {len(lm.depformer_emb)}")

    # Check depformer_norms (CRITICAL: accessed in forward_depformer_training)
    if len(lm.depformer_norms) != 16:
        errors.append(f"depformer_norms: expected 16, got {len(lm.depformer_norms)}")

    # Check depformer layers
    for i, layer in enumerate(lm.depformer.layers):
        # Check in_projs ModuleList
        if hasattr(layer.self_attn, 'in_projs'):
            if len(layer.self_attn.in_projs) != 16:
                errors.append(f"depformer.layers[{i}].self_attn.in_projs: expected 16, got {len(layer.self_attn.in_projs)}")

        # Check out_projs ModuleList
        if hasattr(layer.self_attn, 'out_projs'):
            if len(layer.self_attn.out_projs) != 16:
                errors.append(f"depformer.layers[{i}].self_attn.out_projs: expected 16, got {len(layer.self_attn.out_projs)}")

        # Check gating
        if len(layer.gating) != 16:
            errors.append(f"depformer.layers[{i}].gating: expected 16, got {len(layer.gating)}")

    # Check linears
    if len(lm.linears) != 16:
        errors.append(f"linears: expected 16, got {len(lm.linears)}")

    if errors:
        raise ValueError(
            f"Model validation failed for user stream extension:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return True


def validate_original_model(lm: LMModel) -> bool:
    """
    Validate that a model has original (non-extended) architecture.

    Checks:
        - depformer_in has 8 embeddings
        - depformer_emb has 7 embeddings
        - depformer_norms has 8 LayerNorm modules
        - All depformer layers have:
            - 8 in_projs Linear layers
            - 8 out_projs Linear layers
            - 8 gating modules
        - linears has 8 output projections

    Args:
        lm: LMModel to validate

    Returns:
        True if model has original architecture, False otherwise

    Raises:
        ValueError: If validation fails with detailed error message
    """
    errors = []

    # Check depformer_in
    if len(lm.depformer_in) != 8:
        errors.append(f"depformer_in: expected 8, got {len(lm.depformer_in)}")

    # Check depformer_emb
    if len(lm.depformer_emb) != 7:
        errors.append(f"depformer_emb: expected 7, got {len(lm.depformer_emb)}")

    # Check depformer_norms
    if len(lm.depformer_norms) != 8:
        errors.append(f"depformer_norms: expected 8, got {len(lm.depformer_norms)}")

    # Check depformer layers
    for i, layer in enumerate(lm.depformer.layers):
        # Check in_projs ModuleList
        if hasattr(layer.self_attn, 'in_projs'):
            if len(layer.self_attn.in_projs) != 8:
                errors.append(f"depformer.layers[{i}].self_attn.in_projs: expected 8, got {len(layer.self_attn.in_projs)}")

        # Check out_projs ModuleList
        if hasattr(layer.self_attn, 'out_projs'):
            if len(layer.self_attn.out_projs) != 8:
                errors.append(f"depformer.layers[{i}].self_attn.out_projs: expected 8, got {len(layer.self_attn.out_projs)}")

        # Check gating
        if len(layer.gating) != 8:
            errors.append(f"depformer.layers[{i}].gating: expected 8, got {len(layer.gating)}")

    # Check linears
    if len(lm.linears) != 8:
        errors.append(f"linears: expected 8, got {len(lm.linears)}")

    if errors:
        raise ValueError(
            f"Model validation failed for original architecture:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return True


def get_model_architecture_info(lm: LMModel) -> dict:
    """
    Get detailed architecture information from an LMModel.

    Args:
        lm: LMModel to inspect

    Returns:
        Dictionary with architecture details
    """
    # Get attention module info from first depformer layer if available
    in_projs_count = None
    out_projs_count = None
    gating_count = None

    if hasattr(lm, "depformer") and len(lm.depformer.layers) > 0:
        first_layer = lm.depformer.layers[0]
        if hasattr(first_layer.self_attn, "in_projs"):
            in_projs_count = len(first_layer.self_attn.in_projs)
        if hasattr(first_layer.self_attn, "out_projs"):
            out_projs_count = len(first_layer.self_attn.out_projs)
        if hasattr(first_layer, "gating"):
            gating_count = len(first_layer.gating)

    return {
        "n_q": getattr(lm, "n_q", None),
        "dep_q": getattr(lm, "dep_q", None),
        "num_codebooks": getattr(lm, "num_codebooks", None),
        "depformer_in_count": len(lm.depformer_in) if hasattr(lm, "depformer_in") else None,
        "depformer_emb_count": len(lm.depformer_emb) if hasattr(lm, "depformer_emb") else None,
        "depformer_norms_count": len(lm.depformer_norms) if hasattr(lm, "depformer_norms") else None,
        "depformer_layers_count": len(lm.depformer.layers) if hasattr(lm, "depformer") else None,
        "in_projs_count": in_projs_count,
        "out_projs_count": out_projs_count,
        "gating_count": gating_count,
        "linears_count": len(lm.linears) if hasattr(lm, "linears") else None,
        "text_vocab_size": lm.text_emb.weight.size(0) if hasattr(lm, "text_emb") else None,
        "audio_vocab_size": lm.depformer_in[0].weight.size(0) if hasattr(lm, "depformer_in") and len(lm.depformer_in) > 0 else None,
    }
