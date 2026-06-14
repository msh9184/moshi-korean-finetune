# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""Configuration management for Stage 2 synthesis pipeline.

Updated 2026-01-20: Added PersonaPlex-inspired features:
- Back-annotation for real conversation data
- Hybrid data strategy (Real + Synthetic)
- Voice embedding system
- FullDuplexBench-style metrics
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
import yaml


# =============================================================================
# Data Source Configuration (PersonaPlex-inspired Hybrid Strategy)
# =============================================================================

@dataclass
class CorpusSourceConfig:
    """Configuration for a single corpus source."""
    type: str  # "real", "synthetic_assistant", "synthetic_customer_service", "aihub", "nikl", "custom"
    name: str
    path: str
    weight: float = 1.0
    priority: str = "normal"  # "normal", "high"
    is_real_conversation: bool = False  # PersonaPlex: Real vs Synthetic distinction


@dataclass
class BackAnnotationConfig:
    """Back-annotation configuration (PersonaPlex key feature).

    Back-annotation: Generate prompts for existing real conversations
    to enable prompt-based persona learning from natural dialogue.
    """
    enabled: bool = True
    llm_model: str = "qwen2.5-72b-instruct"  # Local LLM for GPU server

    # Detail level distribution (PersonaPlex uses varying detail)
    detail_levels: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.3,    # "You enjoy having a good conversation."
        "medium": 0.4,    # Role + basic context
        "complex": 0.3,   # Full persona with name, background, scenario
    })

    # Pre-generated prompts path (for offline processing)
    pregenerated_prompts_path: Optional[str] = None

    # Output format
    output_format: str = "jsonl"  # jsonl, json

    # Batch processing
    batch_size: int = 50


@dataclass
class HybridDataConfig:
    """Hybrid data strategy configuration (PersonaPlex core strategy).

    PersonaPlex uses:
    - 35% Real conversations (1,217h)
    - 12% Synthetic assistant (410h)
    - 53% Synthetic customer service (1,840h)

    K-Moshi targets:
    - 40% Real Korean conversations
    - 15% Identity Q&A
    - 30% Customer service
    - 15% General dialogue
    """
    enabled: bool = True

    # Target ratios (must sum to 1.0)
    ratios: Dict[str, float] = field(default_factory=lambda: {
        "real_conversation": 0.40,       # KSponSpeech, AI Hub
        "identity_qa": 0.15,             # K-Moshi identity system
        "customer_service": 0.30,        # Banking, medical, etc.
        "general_dialogue": 0.15,        # Daily conversation
    })

    # Total target hours
    total_target_hours: float = 1000.0


@dataclass
class CorpusConfig:
    """Text corpus configuration."""
    sources: List[CorpusSourceConfig] = field(default_factory=list)

    # Hybrid data strategy (PersonaPlex-inspired)
    hybrid: HybridDataConfig = field(default_factory=HybridDataConfig)

    # Back-annotation for real data (PersonaPlex-inspired)
    back_annotation: BackAnnotationConfig = field(default_factory=BackAnnotationConfig)

    # Normalization settings
    remove_special_chars: bool = True
    normalize_numbers: bool = True
    normalize_units: bool = True
    max_turn_length: int = 100  # words

    # Spoken conversion (LLM)
    spoken_conversion_enabled: bool = True
    spoken_conversion_model: str = "qwen2.5-72b-instruct"  # Local LLM
    spoken_conversion_batch_size: int = 100


# =============================================================================
# Identity & Persona Configuration (PersonaPlex-inspired Prompting)
# =============================================================================

@dataclass
class TextPromptTemplateConfig:
    """Text prompt template configuration (PersonaPlex Hybrid Prompting).

    PersonaPlex uses text prompts to define:
    - Role (e.g., customer service agent, QA assistant)
    - Background (character/context)
    - Scenario context

    Examples from PersonaPlex:
    - Generic: "You are a wise and friendly teacher..."
    - Banking: "You work for First Neuron Bank...your name is..."
    - Medical: "You work for Dr. Jones's medical office..."
    """
    # Default K-Moshi prompt
    default_prompt: str = """당신은 케이모시(K-Moshi)입니다. 한국어 AI 음성 비서입니다.

특징:
- 친절하고 따뜻한 말투를 사용합니다
- 존댓말을 기본으로 하되, 자연스럽게 대화합니다
- 사용자의 질문에 명확하고 도움이 되는 답변을 제공합니다
- 모르는 것은 솔직히 모른다고 말합니다

대화 스타일:
- 적절한 맞장구 ("네", "아, 그렇군요", "이해해요")
- 자연스러운 반응 ("음...", "글쎄요")
- 공감 표현 ("힘드셨겠네요", "좋은 소식이네요")"""

    # Customer service templates
    customer_service_templates_dir: str = "identity/data/customer_service_prompts/"

    # Enable scenario-specific prompts
    scenario_prompts_enabled: bool = True


@dataclass
class IdentityConfig:
    """K-Moshi identity system configuration."""
    name: str = "K-Moshi"
    aliases: List[str] = field(default_factory=lambda: ["케이모시", "모시", "K모시"])

    # Personality traits
    traits: List[str] = field(default_factory=list)
    speaking_style: str = "존댓말 기본"

    # Q&A generation
    qa_templates_enabled: bool = True
    qa_categories: List[str] = field(default_factory=lambda: [
        "self_introduction",
        "capabilities",
        "limitations",
        "creator_info",
        "technical_info"
    ])
    variations_per_question: int = 5
    include_follow_ups: bool = True

    # Injection rate
    injection_rate: float = 0.15

    # Text prompt templates (PersonaPlex-inspired)
    text_prompts: TextPromptTemplateConfig = field(default_factory=TextPromptTemplateConfig)


@dataclass
class TurnTakingConfig:
    """Turn-taking timing configuration."""
    min_gap: float = 0.1
    max_gap: float = 1.5
    mean_gap: float = 0.4
    gap_distribution: str = "lognormal"


@dataclass
class OverlapConfig:
    """Overlap configuration."""
    enabled: bool = True
    probability: float = 0.35
    min_duration: float = 0.1
    max_duration: float = 1.0
    mean_duration: float = 0.3


@dataclass
class BackchannelConfig:
    """Backchannel configuration."""
    enabled: bool = True
    probability: float = 0.25
    tokens: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"text": "응", "weight": 0.30},
        {"text": "네", "weight": 0.25},
        {"text": "아", "weight": 0.15},
        {"text": "음", "weight": 0.15},
        {"text": "그래", "weight": 0.10},
        {"text": "맞아", "weight": 0.05},
    ])
    offset_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    overlap_with_speaker: bool = True


@dataclass
class BargeInConfig:
    """Barge-in configuration."""
    enabled: bool = True
    probability: float = 0.08
    trigger_conditions: List[str] = field(default_factory=lambda: [
        "urgency_detected",
        "clarification_needed",
        "strong_agreement"
    ])
    cut_previous_turn: bool = True


@dataclass
class TimingConfig:
    """Full-duplex timing configuration."""
    turn_taking: TurnTakingConfig = field(default_factory=TurnTakingConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    backchannel: BackchannelConfig = field(default_factory=BackchannelConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)


@dataclass
class TTSModelConfig:
    """TTS model configuration."""
    type: str  # "openaudio_s1", "supertonic"
    model_path: str
    sample_rate: int = 24000


@dataclass
class TTSStrategyConfig:
    """TTS mixing strategy configuration."""
    moshi_ratio: float = 1.0  # 100% primary for Moshi
    user_ratio: float = 0.7   # 70% primary, 30% secondary for User


@dataclass
class TTSConfig:
    """TTS configuration."""
    primary: TTSModelConfig = field(default_factory=lambda: TTSModelConfig(
        type="openaudio_s1",
        model_path="/models/openaudio-s1-mini"
    ))
    secondary: TTSModelConfig = field(default_factory=lambda: TTSModelConfig(
        type="supertonic",
        model_path="/models/supertonic-2"
    ))
    strategy: TTSStrategyConfig = field(default_factory=TTSStrategyConfig)

    batch_size: int = 32
    num_workers: int = 4

    # Quality control
    min_audio_duration: float = 0.3
    max_audio_duration: float = 30.0
    normalize_volume: bool = True
    target_db: float = -23.0


# =============================================================================
# Voice Embedding System (PersonaPlex-inspired)
# =============================================================================

@dataclass
class VoiceEmbeddingConfig:
    """Voice embedding configuration (PersonaPlex key feature).

    PersonaPlex provides 16 pre-packaged voice embeddings:
    - NAT (Natural): 4F + 4M = 8 voices
    - VAR (Variety): 5F + 5M = 10 voices

    For K-Moshi, we use a single consistent voice embedding.
    """
    enabled: bool = True
    embedding_path: str = ""  # Path to .pt file
    embedding_dim: int = 512  # WavLM-TDNN embedding dimension

    # Speaker similarity verification (WavLM-TDNN)
    verify_similarity: bool = True
    min_similarity_threshold: float = 0.60  # PersonaPlex achieves 0.650


@dataclass
class MoshiVoiceConfig:
    """Moshi voice configuration."""
    reference_audio: str = ""
    voice_id: str = "kmoshi_v1"
    gender: str = "neutral"
    age_group: str = "young_adult"
    tone: str = "warm_professional"

    # Voice embedding (PersonaPlex-inspired)
    embedding: VoiceEmbeddingConfig = field(default_factory=VoiceEmbeddingConfig)

    # Voice consistency across all synthetic data
    enforce_consistency: bool = True


@dataclass
class UserVoiceConfig:
    """User voices configuration."""
    min_voices: int = 10
    max_voices: int = 20
    gender_ratio: float = 0.5  # male:female ratio
    age_groups: List[str] = field(default_factory=lambda: ["young", "middle", "senior"])
    regional_accents: List[str] = field(default_factory=lambda: ["seoul", "gyeongsang", "jeolla"])
    reference_dir: str = ""

    # User voice embeddings (multiple)
    embeddings_dir: str = ""  # Directory containing user voice .pt files


@dataclass
class VoiceAssignmentConfig:
    """Voice assignment configuration."""
    mode: str = "random"  # random, round_robin, weighted
    persist_per_dialogue: bool = True


@dataclass
class VoiceConfig:
    """Voice management configuration."""
    moshi: MoshiVoiceConfig = field(default_factory=MoshiVoiceConfig)
    users: UserVoiceConfig = field(default_factory=UserVoiceConfig)
    assignment: VoiceAssignmentConfig = field(default_factory=VoiceAssignmentConfig)


@dataclass
class WERConfig:
    """WER filtering configuration."""
    enabled: bool = True
    max_wer_moshi: float = 0.15
    max_wer_user: float = 0.25
    samples_per_dialogue: int = 10
    select_best: bool = True


@dataclass
class AudioQualityConfig:
    """Audio quality configuration."""
    min_snr: float = 20.0
    check_clipping: bool = True
    max_silence_ratio: float = 0.3


@dataclass
class DialogueQualityConfig:
    """Dialogue quality configuration."""
    min_turns: int = 2
    max_turns: int = 50
    min_moshi_words: int = 10
    min_user_words: int = 5
    check_coherence: bool = False


@dataclass
class FullDuplexMetricsConfig:
    """FullDuplexBench-style evaluation metrics (PersonaPlex benchmark).

    Metrics from PersonaPlex FullDuplexBench:
    - Pause Handling: TOR (Takeover Rate) - lower is better
    - Backchannel: TOR, Frequency, JSD
    - Smooth Turn Taking: Success Rate, Response Latency
    - User Interruption: Success Rate, Latency, Content Quality
    """
    enabled: bool = True

    # Pause handling targets
    pause_handling_max_tor: float = 0.45  # PersonaPlex: 0.358-0.431

    # Backchannel targets
    backchannel_max_tor: float = 0.30      # PersonaPlex: 0.273
    backchannel_min_frequency: float = 0.03  # PersonaPlex: 0.042

    # Smooth turn taking targets
    turn_taking_min_success: float = 0.85   # PersonaPlex: 0.908
    turn_taking_max_latency: float = 0.25   # PersonaPlex: 0.170s

    # User interruption targets
    interruption_min_success: float = 0.90   # PersonaPlex: 0.950
    interruption_max_latency: float = 0.30   # PersonaPlex: 0.240s

    # Speaker similarity (voice consistency)
    min_speaker_similarity: float = 0.60     # PersonaPlex: 0.650


@dataclass
class QualityConfig:
    """Quality assurance configuration."""
    wer: WERConfig = field(default_factory=WERConfig)
    audio: AudioQualityConfig = field(default_factory=AudioQualityConfig)
    dialogue: DialogueQualityConfig = field(default_factory=DialogueQualityConfig)

    # FullDuplexBench-style metrics (PersonaPlex-inspired)
    full_duplex: FullDuplexMetricsConfig = field(default_factory=FullDuplexMetricsConfig)


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "/path/to/data"

    audio_format: str = "wav"  # wav or flac
    sample_rate: int = 24000
    channels: int = 2  # Stereo

    audio_dir: str = "audio"
    alignment_dir: str = "alignments"
    manifest_file: str = "manifest.jsonl"


@dataclass
class DistributedConfig:
    """Distributed processing configuration."""
    num_machines: int = 1
    machine_id: int = 0
    sharding: str = "round_robin"


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    num_workers: int = 8
    batch_size: int = 32

    checkpoint_interval: int = 1000
    resume_from_checkpoint: bool = True

    log_interval: int = 100
    verbose: bool = True


@dataclass
class Stage2Config:
    """Complete Stage 2 pipeline configuration."""
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Stage2Config":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Stage2Config":
        """Create config from dictionary."""
        config = cls()

        # Parse corpus sources
        if "corpus" in data:
            corpus_data = data["corpus"]
            config.corpus.sources = [
                CorpusSourceConfig(**src)
                for src in corpus_data.get("sources", [])
            ]
            if "normalization" in corpus_data:
                for k, v in corpus_data["normalization"].items():
                    if hasattr(config.corpus, k):
                        setattr(config.corpus, k, v)
            if "spoken_conversion" in corpus_data:
                config.corpus.spoken_conversion_enabled = corpus_data["spoken_conversion"].get("enabled", True)
                config.corpus.spoken_conversion_model = corpus_data["spoken_conversion"].get("llm_model", "gemma-2-27b")

        # Parse identity
        if "identity" in data:
            id_data = data["identity"]
            config.identity.name = id_data.get("name", "K-Moshi")
            config.identity.aliases = id_data.get("aliases", [])
            if "personality" in id_data:
                config.identity.traits = id_data["personality"].get("traits", [])
                config.identity.speaking_style = id_data["personality"].get("speaking_style", "")
            if "qa_templates" in id_data:
                config.identity.qa_templates_enabled = id_data["qa_templates"].get("enabled", True)
                config.identity.qa_categories = id_data["qa_templates"].get("categories", [])
            config.identity.injection_rate = id_data.get("injection_rate", 0.15)

        # Parse timing
        if "timing" in data:
            timing_data = data["timing"]
            if "turn_taking" in timing_data:
                for k, v in timing_data["turn_taking"].items():
                    if hasattr(config.timing.turn_taking, k):
                        setattr(config.timing.turn_taking, k, v)
            if "overlap" in timing_data:
                for k, v in timing_data["overlap"].items():
                    if hasattr(config.timing.overlap, k):
                        setattr(config.timing.overlap, k, v)
            if "backchannel" in timing_data:
                for k, v in timing_data["backchannel"].items():
                    if hasattr(config.timing.backchannel, k):
                        setattr(config.timing.backchannel, k, v)
            if "barge_in" in timing_data:
                for k, v in timing_data["barge_in"].items():
                    if hasattr(config.timing.barge_in, k):
                        setattr(config.timing.barge_in, k, v)

        # Parse TTS
        if "tts" in data:
            tts_data = data["tts"]
            if "primary" in tts_data:
                config.tts.primary = TTSModelConfig(**tts_data["primary"])
            if "secondary" in tts_data:
                config.tts.secondary = TTSModelConfig(**tts_data["secondary"])
            if "strategy" in tts_data:
                config.tts.strategy.moshi_ratio = tts_data["strategy"].get("moshi_ratio", 1.0)
                config.tts.strategy.user_ratio = tts_data["strategy"].get("user_ratio", 0.7)
            config.tts.batch_size = tts_data.get("batch_size", 32)

        # Parse voice
        if "voice" in data:
            voice_data = data["voice"]
            if "moshi" in voice_data:
                for k, v in voice_data["moshi"].items():
                    if hasattr(config.voice.moshi, k):
                        setattr(config.voice.moshi, k, v)
            if "users" in voice_data:
                for k, v in voice_data["users"].items():
                    if hasattr(config.voice.users, k):
                        setattr(config.voice.users, k, v)

        # Parse quality
        if "quality" in data:
            q_data = data["quality"]
            if "wer" in q_data:
                for k, v in q_data["wer"].items():
                    if hasattr(config.quality.wer, k):
                        setattr(config.quality.wer, k, v)
            if "audio" in q_data:
                for k, v in q_data["audio"].items():
                    if hasattr(config.quality.audio, k):
                        setattr(config.quality.audio, k, v)

        # Parse output
        if "output" in data:
            for k, v in data["output"].items():
                if hasattr(config.output, k):
                    setattr(config.output, k, v)

        # Parse distributed
        if "distributed" in data:
            for k, v in data["distributed"].items():
                if hasattr(config.distributed, k):
                    setattr(config.distributed, k, v)

        # Parse processing
        if "processing" in data:
            for k, v in data["processing"].items():
                if hasattr(config.processing, k):
                    setattr(config.processing, k, v)

        return config

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        data = self._to_dict()

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _to_dict(self) -> dict:
        """Convert config to dictionary."""
        # Simple implementation - can be enhanced
        import dataclasses
        return dataclasses.asdict(self)


def get_default_config() -> Stage2Config:
    """Get default Stage 2 configuration."""
    return Stage2Config()
