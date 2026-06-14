import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple, Any, List

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask, compute_audio_loss_per_speaker
from .utils import TrainState

logger = logging.getLogger("eval")


# =============================================================================
# EVALUATION WITH SPEAKER CONDITIONING SUPPORT
# =============================================================================
# This module extends the evaluation pipeline to support:
# 1. Speaker embedding extraction from reference audio
# 2. Audio prompting (PersonaPlex style) with deterministic sampling
# 3. Reference metadata saving for reproducibility analysis
#
# Key Design Principles:
# - DETERMINISTIC: Same input always produces same reference selection
# - FSDP-COMPATIBLE: All ranks participate in model forward together
# - REPRODUCIBLE: Full metadata saved for experiment reproduction
# =============================================================================


@dataclass
class EvalSpeakerConditioningInfo:
    """Information about speaker conditioning used during evaluation.

    This dataclass stores all relevant information about the speaker
    conditioning applied during evaluation, enabling:
    - Reproducibility analysis
    - Debug logging
    - Sample saving with full context

    Supports both single-sample and batch-level information:
    - Single-sample fields (legacy): reference_text, reference_start_sec, etc.
    - Batch-level fields (new): reference_texts, reference_start_secs, etc.

    When batch fields are available, sample_saver uses batch_idx to get per-sample info.
    Otherwise, it falls back to single-sample fields for backward compatibility.

    Attributes:
        enabled: Whether speaker conditioning was active
        method: Conditioning method ("encoder", "prompt", "both")
        speaker_embedding: Extracted speaker embedding [B, D_spk] (if encoder used)
        reference_audio: Raw reference audio [B, T] (if available)
        reference_audio_sample_rate: Sample rate of reference_audio (16000 or 24000)

        # Single-sample legacy fields (for backward compatibility)
        reference_start_sec: Start time of reference in original audio
        reference_end_sec: End time of reference in original audio
        reference_duration_sec: Actual duration of reference audio in seconds
        reference_text: Transcription of reference (if available)
        reference_num_frames: Number of audio frames in reference
        source_file: Original audio file path (for train split tracking)

        # Batch-level fields (new - preferred when available)
        reference_texts: List of reference texts, one per batch item
        reference_start_secs: List of start times, one per batch item
        reference_end_secs: List of end times, one per batch item
        source_files: List of source file paths, one per batch item

        prompt_mask: Mask indicating prompt positions [B, T] (if prompting used)
        sampling_strategy: Strategy used for reference selection
        deterministic: Whether deterministic sampling was used
    """
    enabled: bool = False
    method: str = "none"
    speaker_embedding: Optional[torch.Tensor] = None
    reference_audio: Optional[torch.Tensor] = None
    reference_audio_sample_rate: int = 24000  # Sample rate of reference_audio

    # Single-sample legacy fields (for backward compatibility)
    reference_start_sec: float = 0.0
    reference_end_sec: float = 0.0
    reference_duration_sec: float = 0.0  # Computed actual duration
    reference_text: Optional[str] = None
    reference_num_frames: int = 0  # Number of audio frames
    source_file: Optional[str] = None  # Original audio file for tracking

    # Batch-level fields (new - preferred when available)
    reference_texts: Optional[List[str]] = None  # Per-batch-item texts
    reference_start_secs: Optional[List[float]] = None  # Per-batch-item start times
    reference_end_secs: Optional[List[float]] = None  # Per-batch-item end times
    source_files: Optional[List[str]] = None  # Per-batch-item source files

    prompt_mask: Optional[torch.Tensor] = None
    sampling_strategy: str = "start"
    deterministic: bool = True
    fixed_duration_sec: float = 10.0


def _pad_codes_for_model(codes: torch.Tensor, target_codebooks: int, zero_token_id: int) -> torch.Tensor:
    """Pad codes to match model's expected number of codebooks.

    Args:
        codes: Input codes tensor [B, K, T]
        target_codebooks: Number of codebooks the model expects (e.g., 17)
        zero_token_id: Special token for no-input positions (-1)

    Returns:
        Padded codes tensor [B, target_codebooks, T]
    """
    B, K, T = codes.shape
    if K == target_codebooks:
        return codes
    if K > target_codebooks:
        raise ValueError(
            f"Data has {K} codebooks but model expects only {target_codebooks}."
        )
    pad_amount = target_codebooks - K
    return torch.nn.functional.pad(codes, (0, 0, 0, pad_amount), value=zero_token_id)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


@dataclass
class EvalReturnData:
    """Return data from evaluation containing all tensors and metadata.

    This dataclass provides clear separation between:
    - original_codes: Original codes before prompting (for sample saving)
    - prompted_codes: Codes with prompt prepended (for loss/metric computation)
    - output: Model output (based on prompted_codes)

    This distinction is CRITICAL when audio_prompt_module prepends reference frames.
    """
    original_codes: Optional[torch.Tensor] = None  # [B, K, T] - original
    prompted_codes: Optional[torch.Tensor] = None  # [B, K, T+P] - with prompt prefix
    output: Optional[Any] = None
    user_text_alignments: Optional[list] = None
    moshi_text_raw_list: Optional[list] = None
    audio_paths: Optional[list] = None
    speaker_conditioning_info: Optional[EvalSpeakerConditioningInfo] = None


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
    speaker_encoder: Optional[Any] = None,
    audio_prompt_module: Optional[Any] = None,
    mimi: Optional[Any] = None,
    text_tokenizer: Optional[Any] = None,
) -> EvalReturnData:
    """
    Evaluate model on validation data with optional speaker conditioning.

    This function supports the full speaker conditioning pipeline during evaluation:
    1. Deterministic reference sampling (no randomness)
    2. Speaker embedding extraction via encoder
    3. Audio/text prompting (PersonaPlex style)

    Args:
        model: The FSDP-wrapped model
        eval_data_loader: Iterator over validation batches
        state: Training state to update with eval metrics
        args: Training arguments including eval_samples config
        speaker_encoder: Optional speaker encoder for embedding extraction
        audio_prompt_module: Optional audio prompt module for PersonaPlex prompting
        mimi: Optional Mimi codec for decoding reference audio from codes
        text_tokenizer: Optional text tokenizer for decoding reference text

    Returns:
        EvalReturnData containing:
            - original_codes: Original codes [B, K, T] for sample saving
            - prompted_codes: Codes with prompt [B, K, T+P] for metric computation
            - output: Model output (based on prompted_codes)
            - user_text_alignments: User text alignments from last batch
            - moshi_text_raw_list: Moshi text from last batch
            - audio_paths: Audio file paths from last batch
            - speaker_conditioning_info: Speaker conditioning metadata

        Returns EvalReturnData with all None fields if no batches were processed.

    Speaker Conditioning Protocol:
        1. Reference is sampled DETERMINISTICALLY (sample_strategy="start")
        2. Fixed duration (config.fixed_duration_sec) is used
        3. Same input always produces same reference selection
        4. Full metadata saved in EvalSpeakerConditioningInfo for reproducibility
    """
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)

    # Use configured eval_samples, default to 100 if not set
    max_samples_per_rank = getattr(args, "eval_samples", 100) // get_world_size()
    max_samples_per_rank = max(max_samples_per_rank, 1)  # At least 1 sample per rank

    # Check training mode configuration
    enable_user_stream = getattr(args.korean, 'enable_user_stream', False)
    full_duplex_input = getattr(args.korean, 'full_duplex_input', True)
    use_stereo_data = enable_user_stream or full_duplex_input

    main_logger_info(f"[EVAL] Starting evaluation with max {max_samples_per_rank} samples per rank")
    if enable_user_stream:
        main_logger_info(f"[EVAL] USER-STREAM mode (dep_q={model.dep_q})")
    elif full_duplex_input:
        main_logger_info(f"[EVAL] FULL-DUPLEX mode (dep_q={model.dep_q}, user audio as context)")

    # =========================================================================
    # SPEAKER CONDITIONING SETUP FOR EVALUATION
    # =========================================================================
    speaker_conditioning_enabled = (
        hasattr(args, 'speaker') and
        getattr(args.speaker, 'enabled', False)
    )
    speaker_method = getattr(args.speaker, 'method', 'none') if speaker_conditioning_enabled else 'none'

    if speaker_conditioning_enabled:
        main_logger_info(
            f"[EVAL] Speaker conditioning enabled: method={speaker_method}, "
            f"encoder={'provided' if speaker_encoder else 'none'}, "
            f"prompt_module={'provided' if audio_prompt_module else 'none'}"
        )

    # Speaker conditioning info for the last batch (for sample saving)
    last_speaker_conditioning_info = EvalSpeakerConditioningInfo(
        enabled=speaker_conditioning_enabled,
        method=speaker_method,
        deterministic=True,  # Always deterministic in eval
    )

    text_loss = torch.tensor(0.0).cuda()
    audio_loss = torch.tensor(0.0).cuda()
    moshi_audio_loss = torch.tensor(0.0).cuda()
    user_audio_loss = torch.tensor(0.0).cuda()
    last_original_codes = None  # Original codes without prompt (for sample saving)
    last_prompted_codes = None  # Codes with prompt prefix (for metrics)
    last_output = None
    last_user_text_alignments = None
    last_moshi_text_raw_list = None
    last_audio_paths = None  # Track audio paths for dialogue saving
    model.eval()

    # =========================================================================
    # CRITICAL FIX v2: FSDP-compatible synchronized evaluation loop
    # =========================================================================
    # Problem 1: Different ranks may have different amounts of eval data due to
    # uneven data distribution or empty alignments.
    #
    # Problem 2 (CRITICAL): FSDP model forward pass requires ALL ranks to
    # participate simultaneously. If only some ranks call model(), the others
    # hang waiting for FSDP's internal all_gather operations → NCCL deadlock!
    #
    # Solution: Use "ALL ranks must have data" strategy instead of "ANY rank
    # has data". This ensures all ranks either process a batch together or
    # stop together. Some data may be dropped, but FSDP synchronization is
    # maintained.
    # =========================================================================

    eval_iterator = iter(eval_data_loader)
    while True:
        # Check if this rank has more data
        try:
            batch = next(eval_iterator)
            has_data = torch.tensor([1], device="cuda", dtype=torch.long)
        except StopIteration:
            has_data = torch.tensor([0], device="cuda", dtype=torch.long)
            batch = None

        # Check if this rank should stop due to max_samples limit
        if num_samples >= max_samples_per_rank:
            has_data = torch.tensor([0], device="cuda", dtype=torch.long)
            batch = None  # Discard batch if limit reached

        # Synchronize: gather has_data from all ranks
        all_has_data = [torch.zeros_like(has_data) for _ in range(get_world_size())]
        dist.all_gather(all_has_data, has_data)

        # =====================================================================
        # CRITICAL: ALL ranks must have data to continue
        # =====================================================================
        # If ANY rank has no data, ALL ranks must stop. This ensures:
        # 1. All ranks call model() together (FSDP requirement)
        # 2. No rank is left waiting for others at model() call
        # 3. NCCL operations stay synchronized across all ranks
        #
        # Trade-off: Some batches may be dropped from ranks that have extra
        # data. This is acceptable for evaluation purposes.
        # =====================================================================
        all_have_data = all(t.item() == 1 for t in all_has_data)
        if not all_have_data:
            # At least one rank has no data - ALL ranks must stop
            main_logger_info(
                f"[EVAL] Stopping: rank {get_rank()} has_data={has_data.item()}, "
                f"all_has_data={[t.item() for t in all_has_data]}"
            )
            break

        # All ranks have data - safe to process together
        last_user_text_alignments = batch.user_text_alignments
        last_moshi_text_raw_list = batch.moshi_text_raw_list
        last_audio_paths = batch.audio_paths  # Track for dialogue saving
        num_samples += 1

        with torch.no_grad():
            codes = batch.codes
            # Pad codes from 9 to 17 codebooks for moshiko model compatibility
            # Stereo modes (USER-STREAM, FULL-DUPLEX) have 17 codebooks, no padding needed
            # MONOLOGUE mode has 9 codebooks, needs padding to 17
            if not use_stereo_data:
                codes = _pad_codes_for_model(codes, model.num_codebooks, model.zero_token_id)

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # =================================================================
            # SPEAKER CONDITIONING FOR EVALUATION
            # =================================================================
            # Use DETERMINISTIC sampling for reproducible evaluation
            # Same input → Same reference → Same output
            # =================================================================
            speaker_embedding = None
            prompt_mask = None
            prompted_codes = codes

            if speaker_conditioning_enabled:
                # Get reference audio from batch (if available)
                reference_audio = getattr(batch, 'speaker_reference_audio', None)

                # Extract speaker embedding (encoder method)
                if speaker_encoder is not None and reference_audio is not None:
                    # NOTE: Use codes.device instead of model.device
                    # FSDP-wrapped LMModelWrapper doesn't expose .device attribute
                    speaker_embedding = speaker_encoder(
                        reference_audio.to(codes.device),
                        lengths=None,  # Assume all same length
                    )
                    last_speaker_conditioning_info.speaker_embedding = speaker_embedding
                    last_speaker_conditioning_info.reference_audio = reference_audio

                # Apply audio prompting with DETERMINISTIC sampling
                prompt_length = 0  # Track prompt length for loss computation
                if audio_prompt_module is not None:
                    prompted_codes, prompt_mask, prompt_samples = audio_prompt_module(
                        codes,
                        exclude_start=None,  # No exclusion for eval
                        exclude_end=None,
                        deterministic=True,  # CRITICAL: Always deterministic for eval
                    )
                    last_speaker_conditioning_info.prompt_mask = prompt_mask
                    if prompt_samples:
                        # =========================================================
                        # CRITICAL FIX: Store batch-level lists for per-sample metadata
                        # =========================================================
                        # Each item in prompt_samples corresponds to one batch item.
                        # We must store all items' info to enable per-sample metadata
                        # when saving samples (sample_saver uses batch_idx).
                        # =========================================================
                        batch_size = len(prompt_samples)
                        frame_rate = audio_prompt_module.sampler.frame_rate

                        # Initialize batch-level lists
                        reference_start_secs = []
                        reference_end_secs = []
                        reference_texts = []

                        for ps in prompt_samples:
                            reference_start_secs.append(ps.start_idx / frame_rate)
                            reference_end_secs.append(ps.end_idx / frame_rate)

                        # Store batch-level timing info
                        last_speaker_conditioning_info.reference_start_secs = reference_start_secs
                        last_speaker_conditioning_info.reference_end_secs = reference_end_secs

                        # Also store first sample's info for legacy compatibility
                        sample = prompt_samples[0]
                        last_speaker_conditioning_info.reference_start_sec = (
                            sample.start_idx / frame_rate
                        )
                        last_speaker_conditioning_info.reference_end_sec = (
                            sample.end_idx / frame_rate
                        )
                        # Calculate prompt length (frames prepended)
                        prompt_length = sample.end_idx - sample.start_idx

                        # Store source_files if available from batch
                        if batch.audio_paths is not None:
                            last_speaker_conditioning_info.source_files = batch.audio_paths
                            if batch.audio_paths:
                                last_speaker_conditioning_info.source_file = batch.audio_paths[0]

                        # =============================================================
                        # DECODE REFERENCE AUDIO FROM PROMPT CODES FOR SAMPLE SAVING
                        # =============================================================
                        # prompt_samples contains audio codes [8, T_prompt], but for
                        # saving reference.wav we need the raw audio waveform.
                        # Use Mimi codec to decode the audio codes back to waveform.
                        # =============================================================
                        if mimi is not None and sample.audio_codes is not None:
                            try:
                                # sample.audio_codes is [8, T_prompt]
                                # Mimi expects [B, K, T] format
                                # NOTE: Use codes.device instead of model.device
                                # FSDP-wrapped LMModelWrapper doesn't expose .device attribute
                                device = codes.device
                                ref_codes = sample.audio_codes.unsqueeze(0).to(device)
                                ref_codes = ref_codes.clamp(0, 2047)  # Clamp to valid range

                                # Decode using Mimi
                                with torch.no_grad():
                                    ref_audio = mimi.decode(ref_codes)  # [1, 1, T_audio]
                                    # Reshape to [1, T_audio] for storage
                                    ref_audio = ref_audio.squeeze(0)  # [1, T_audio]

                                # Store reference audio for each batch item
                                # For now, we use the first sample's reference for all
                                # batch items (since we're using last_speaker_conditioning_info)
                                batch_size = codes.shape[0]
                                # Replicate for batch dimension: [B, T_audio]
                                last_speaker_conditioning_info.reference_audio = ref_audio.expand(
                                    batch_size, -1
                                ).clone()

                                # CRITICAL: Set sample rate to 24000 (Mimi output)
                                # This is essential for correct playback of saved audio
                                last_speaker_conditioning_info.reference_audio_sample_rate = 24000
                                last_speaker_conditioning_info.reference_num_frames = ref_audio.shape[-1]
                                last_speaker_conditioning_info.reference_duration_sec = (
                                    ref_audio.shape[-1] / 24000.0
                                )

                                main_logger_info(
                                    f"[EVAL] Decoded reference audio: "
                                    f"codes={list(sample.audio_codes.shape)}, "
                                    f"audio={list(ref_audio.shape)}, "
                                    f"sample_rate=24000Hz, "
                                    f"duration={ref_audio.shape[-1] / 24000:.2f}s"
                                )
                            except Exception as e:
                                main_logger_info(
                                    f"[EVAL] Warning: Failed to decode reference audio: {e}"
                                )

                        # =========================================================
                        # DECODE REFERENCE TEXT FOR ALL BATCH ITEMS
                        # =========================================================
                        # Decode text from each prompt_sample to get per-sample
                        # reference texts. This enables correct metadata saving.
                        # =========================================================
                        if text_tokenizer is not None:
                            reference_texts = []
                            for ps in prompt_samples:
                                if ps.text_tokens is not None:
                                    try:
                                        # Filter out padding tokens (0, 3, 32000)
                                        valid_tokens = [
                                            int(t) for t in ps.text_tokens.tolist()
                                            if int(t) not in {0, 3, 32000} and int(t) >= 0
                                        ]
                                        if valid_tokens:
                                            ref_text = text_tokenizer.decode(valid_tokens)
                                            reference_texts.append(ref_text)
                                        else:
                                            reference_texts.append("")
                                    except Exception as e:
                                        main_logger_info(
                                            f"[EVAL] Warning: Failed to decode reference text: {e}"
                                        )
                                        reference_texts.append("")
                                else:
                                    reference_texts.append("")

                            # Store batch-level reference texts
                            last_speaker_conditioning_info.reference_texts = reference_texts
                            # Also store first sample's text for legacy compatibility
                            if reference_texts:
                                last_speaker_conditioning_info.reference_text = reference_texts[0]
                                if reference_texts[0]:
                                    main_logger_info(
                                        f"[EVAL] Reference texts: {len(reference_texts)} items, "
                                        f"first=\"{reference_texts[0][:50]}...\""
                                    )

            # FSDP: All ranks call model() together
            output = model(
                codes=prompted_codes,
                condition_tensors=condition_tensors,
                speaker_embedding=speaker_embedding,  # NEW: Speaker conditioning
            )

            # =================================================================
            # CRITICAL FIX: Use prompted_codes for loss computation
            # =================================================================
            # When audio_prompt_module is used, prompted_codes has reference
            # frames prepended. We must use prompted_codes (not original codes)
            # for loss computation to match output.text_mask dimensions.
            #
            # The prompt_mask indicates which positions are prompt (True) vs
            # target (False). Loss is computed on ALL positions but prompt
            # positions are already masked out by the model's output.mask.
            # =================================================================
            loss_codes = prompted_codes  # Use prompted codes for loss

            text_loss += compute_loss_with_mask(
                output.text_logits,
                loss_codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
                prompt_mask=prompt_mask,  # Exclude prompt positions from loss
            )

            # Compute audio loss with per-speaker breakdown for user stream mode
            audio_codes = loss_codes[:, model.audio_offset : model.audio_offset + model.dep_q]

            if enable_user_stream and model.dep_q == 16:
                # Full duplex mode: compute per-speaker losses
                audio_loss_result = compute_audio_loss_per_speaker(
                    output.logits,
                    audio_codes,
                    output.mask,
                    dep_q=model.dep_q,
                    semantic_weight=args.first_codebook_weight_multiplier,
                    acoustic_weight=1.0,
                    user_semantic_weight=args.user_semantic_weight,
                    user_acoustic_weight=args.user_acoustic_weight,
                    prompt_mask=prompt_mask,  # Exclude prompt positions from loss
                )
                audio_loss += audio_loss_result.total_loss
                moshi_audio_loss += audio_loss_result.moshi_total_loss
                if audio_loss_result.user_total_loss is not None:
                    user_audio_loss += audio_loss_result.user_total_loss
            else:
                # Standard mono mode
                audio_loss += compute_loss_with_mask(
                    output.logits,
                    audio_codes,
                    output.mask,
                    mode="audio",
                    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                    prompt_mask=prompt_mask,  # Exclude prompt positions from loss
                )

            # Store last batch for sample saving and metric computation
            # CRITICAL: Store BOTH original and prompted codes
            # - original_codes: For sample saving (GT vs Prediction comparison)
            # - prompted_codes: For metric computation (shape matches output.mask)
            last_original_codes = codes
            last_prompted_codes = prompted_codes
            last_output = output

    # End of synchronized evaluation loop
    eval_loss = text_loss + audio_loss

    # Barrier to ensure all ranks have finished the evaluation loop
    dist.barrier()
    main_logger_info(f"[EVAL] Rank {get_rank()} processed {num_samples.item()} samples")

    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())

    # Handle edge case: no samples processed
    if total_num_samples == 0:
        main_logger_info("[EVAL] Warning: No samples processed during evaluation!")
        state.this_eval_loss = 0.0
        state.this_eval_perplexity = 1.0
        state.this_audio_loss = 0.0
        state.this_text_loss = 0.0
        model.train()
        return EvalReturnData()

    main_logger_info(f"[EVAL] Finished! Total samples across all ranks: {total_num_samples}")

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)

    # Also reduce user stream losses if applicable
    if enable_user_stream:
        dist.all_reduce(moshi_audio_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(user_audio_loss, op=dist.ReduceOp.SUM)
        moshi_audio_loss /= total_num_samples
        user_audio_loss /= total_num_samples

    text_loss /= total_num_samples
    audio_loss /= total_num_samples
    eval_loss /= total_num_samples

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()
    state.this_audio_loss = audio_loss.item()
    state.this_text_loss = text_loss.item()

    # Store user stream losses in state for logging
    if enable_user_stream:
        state.this_moshi_audio_loss = moshi_audio_loss.item()
        state.this_user_audio_loss = user_audio_loss.item()
        main_logger_info(
            f"[EVAL] Completed: {total_num_samples} total samples, loss={eval_loss.item():.4f}, "
            f"moshi_audio={moshi_audio_loss.item():.4f}, user_audio={user_audio_loss.item():.4f}"
        )
    else:
        main_logger_info(f"[EVAL] Completed: {total_num_samples} total samples, loss={eval_loss.item():.4f}")

    # Log speaker conditioning info
    if speaker_conditioning_enabled:
        main_logger_info(
            f"[EVAL] Speaker conditioning: method={speaker_method}, "
            f"ref_start={last_speaker_conditioning_info.reference_start_sec:.2f}s, "
            f"ref_end={last_speaker_conditioning_info.reference_end_sec:.2f}s"
        )

    # train mode!
    model.train()

    # Return last batch data for optional sample saving (including speaker conditioning info)
    # CRITICAL: Return BOTH original and prompted codes for different use cases:
    # - original_codes: For sample saving (GT audio generation)
    # - prompted_codes: For enhanced evaluation metrics (shape matches output.mask)
    return EvalReturnData(
        original_codes=last_original_codes,
        prompted_codes=last_prompted_codes,
        output=last_output,
        user_text_alignments=last_user_text_alignments,
        moshi_text_raw_list=last_moshi_text_raw_list,
        audio_paths=last_audio_paths,
        speaker_conditioning_info=last_speaker_conditioning_info,
    )
