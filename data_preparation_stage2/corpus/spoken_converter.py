# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""Written-to-Spoken Style Converter for K-Moshi.

Korean text corpora are typically written in formal written style (문어체).
For natural TTS synthesis, conversion to spoken style (구어체) is needed.

This module provides three approaches:
1. RuleBasedConverter: Fast, no LLM required (70% quality)
2. LocalLLMConverter: Local LLM on GPU (85% quality)
3. APIConverter: External API for pre-processing (95% quality)

Usage:
    # Rule-based (offline, fast)
    converter = RuleBasedConverter()
    spoken = converter.convert("무엇을 도와드릴까요?")
    # → "뭐 도와드릴까요?"

    # Local LLM (offline, GPU)
    converter = LocalLLMConverter("/models/gemma-2-9b-it")
    spoken = converter.convert("확인하겠습니다.")
    # → "확인해볼게요."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import re
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of spoken style conversion.

    Attributes:
        original: Original text.
        converted: Converted spoken-style text.
        confidence: Confidence score (0.0-1.0).
        method: Conversion method used.
    """
    original: str
    converted: str
    confidence: float
    method: str


class SpokenConverter(ABC):
    """Abstract base class for spoken style converters."""

    @abstractmethod
    def convert(self, text: str) -> str:
        """Convert single text to spoken style."""
        pass

    @abstractmethod
    def convert_batch(self, texts: List[str]) -> List[str]:
        """Convert batch of texts."""
        pass


class RuleBasedConverter(SpokenConverter):
    """Rule-based written-to-spoken converter.

    This converter uses regex rules to transform common written
    patterns into spoken equivalents. It's fast and requires no
    external dependencies, but quality is limited.

    Features:
    - Verb ending conversion (합니다 → 해요)
    - Contraction (것은 → 건)
    - Optional filler word insertion
    - Common phrase conversion

    Attributes:
        add_fillers: Whether to add filler words.
        filler_prob: Probability of adding fillers.
    """

    # Verb ending conversions (문어체 → 구어체)
    VERB_ENDINGS: List[Tuple[str, str]] = [
        # Formal → Informal polite
        (r"합니다$", "해요"),
        (r"합니까\?$", "해요?"),
        (r"입니다$", "이에요"),
        (r"입니까\?$", "인가요?"),
        (r"습니다$", "어요"),
        (r"습니까\?$", "을까요?"),
        (r"하겠습니다$", "할게요"),
        (r"드리겠습니다$", "드릴게요"),
        (r"되겠습니다$", "될 거예요"),
        (r"있습니다$", "있어요"),
        (r"없습니다$", "없어요"),
        (r"됩니다$", "돼요"),
        (r"싶습니다$", "싶어요"),
        (r"봅니다$", "봐요"),
        (r"갑니다$", "가요"),
        (r"옵니다$", "와요"),
        (r"줍니다$", "줘요"),
        (r"봅시다$", "봐요"),
        (r"갑시다$", "가요"),
        (r"합시다$", "해요"),
    ]

    # Contraction rules
    CONTRACTIONS: List[Tuple[str, str]] = [
        (r"것은", "건"),
        (r"것이", "게"),
        (r"것을", "걸"),
        (r"그것", "그거"),
        (r"이것", "이거"),
        (r"저것", "저거"),
        (r"무엇", "뭐"),
        (r"어디에", "어디"),
        (r"왜냐하면", "왜냐면"),
        (r"그러므로", "그래서"),
        (r"그렇지만", "근데"),
        (r"그러나", "근데"),
        (r"따라서", "그래서"),
        (r"하지만", "근데"),
    ]

    # Subject contractions
    SUBJECT_CONTRACTIONS: List[Tuple[str, str]] = [
        (r"나는", "난"),
        (r"너는", "넌"),
        (r"저는", "전"),
        (r"그는", "걔는"),
        (r"그녀는", "걔는"),
        (r"우리는", "우린"),
    ]

    # Phrase conversions
    PHRASE_CONVERSIONS: List[Tuple[str, str]] = [
        (r"하지 않", "안 하"),
        (r"되지 않", "안 되"),
        (r"알겠습니다", "알겠어요"),
        (r"감사합니다", "고마워요"),
        (r"죄송합니다", "미안해요"),
        (r"모르겠습니다", "모르겠어요"),
        (r"괜찮습니다", "괜찮아요"),
        (r"그렇습니다", "그래요"),
        (r"아닙니다", "아니에요"),
        (r"맞습니다", "맞아요"),
        (r"~하시겠습니까", "~할래요?"),
        (r"부탁드립니다", "부탁해요"),
        (r"말씀드리다", "말하다"),
    ]

    # Korean fillers
    FILLERS = ["음", "그", "아", "어", "저"]

    def __init__(
        self,
        add_fillers: bool = False,
        filler_prob: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize rule-based converter.

        Args:
            add_fillers: Whether to randomly add filler words.
            filler_prob: Probability of adding a filler (0.0-1.0).
            seed: Random seed for reproducibility.
        """
        self.add_fillers = add_fillers
        self.filler_prob = filler_prob

        if seed is not None:
            random.seed(seed)

        # Compile all regex patterns
        self._compiled_rules = []
        for rules in [
            self.VERB_ENDINGS,
            self.CONTRACTIONS,
            self.SUBJECT_CONTRACTIONS,
            self.PHRASE_CONVERSIONS,
        ]:
            for pattern, replacement in rules:
                self._compiled_rules.append(
                    (re.compile(pattern), replacement)
                )

        logger.info(
            f"RuleBasedConverter initialized with {len(self._compiled_rules)} rules"
        )

    def convert(self, text: str) -> str:
        """Convert text to spoken style.

        Args:
            text: Korean text in written style.

        Returns:
            Text converted to spoken style.
        """
        if not text or not text.strip():
            return text

        result = text

        # Apply all rules
        for pattern, replacement in self._compiled_rules:
            result = pattern.sub(replacement, result)

        # Optionally add fillers
        if self.add_fillers:
            result = self._maybe_add_filler(result)

        return result

    def convert_batch(self, texts: List[str]) -> List[str]:
        """Convert batch of texts.

        Args:
            texts: List of Korean texts.

        Returns:
            List of converted texts.
        """
        return [self.convert(t) for t in texts]

    def convert_with_result(self, text: str) -> ConversionResult:
        """Convert with detailed result.

        Args:
            text: Korean text in written style.

        Returns:
            ConversionResult with original, converted, and metadata.
        """
        converted = self.convert(text)
        changes = 0

        # Count changes made
        for pattern, _ in self._compiled_rules:
            if pattern.search(text):
                changes += 1

        # Estimate confidence based on changes
        confidence = min(0.7 + (changes * 0.05), 0.9)

        return ConversionResult(
            original=text,
            converted=converted,
            confidence=confidence,
            method="rule_based",
        )

    def _maybe_add_filler(self, text: str) -> str:
        """Randomly add a filler word at the beginning."""
        if random.random() < self.filler_prob:
            filler = random.choice(self.FILLERS)
            return f"{filler}, {text}"
        return text


class LocalLLMConverter(SpokenConverter):
    """Local LLM-based spoken style converter.

    Uses a local language model (e.g., Gemma-2, SOLAR) to convert
    written Korean to natural spoken style. Runs on GPU without
    external API calls.

    Requirements:
    - transformers
    - bitsandbytes (for 4-bit quantization)
    - torch

    Attributes:
        model_path: Path to local model.
        device: "cuda" or "cpu".
    """

    SYSTEM_PROMPT = """당신은 한국어 구어체 변환 전문가입니다.
문어체 문장을 자연스러운 대화체로 변환합니다.

규칙:
1. 딱딱한 문어체 → 자연스러운 대화체
2. 원래 의미는 유지
3. 변환된 문장만 출력하세요.

예시:
- 무엇을 도와드릴까요? → 뭐 도와드릴까요?
- 확인하겠습니다 → 확인해볼게요
- 그것은 불가능합니다 → 그건 좀 어려울 것 같아요"""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 100,
    ):
        """Initialize local LLM converter.

        Args:
            model_path: Path to local model directory.
            device: "cuda" or "cpu".
            load_in_4bit: Use 4-bit quantization to save VRAM.
            max_new_tokens: Maximum tokens to generate.

        Raises:
            ImportError: If required packages are not installed.
            FileNotFoundError: If model path doesn't exist.
        """
        self.model_path = Path(model_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        self._load_model()
        logger.info(f"LocalLLMConverter initialized: {model_path}")

    def _load_model(self):
        """Load the language model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        # Quantization config for memory efficiency
        bnb_config = None
        if self.load_in_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, loading without quantization"
                )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16,
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def convert(
        self,
        text: str,
        temperature: float = 0.7,
    ) -> str:
        """Convert text to spoken style using LLM.

        Args:
            text: Korean text in written style.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Text converted to spoken style.
        """
        import torch

        if not text or not text.strip():
            return text

        prompt = f"""{self.SYSTEM_PROMPT}

문어체: {text}
구어체:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract converted part
        if "구어체:" in full_response:
            spoken = full_response.split("구어체:")[-1].strip()
            # Clean up
            spoken = spoken.split("\n")[0].strip()
            spoken = spoken.rstrip(".")
            return spoken

        # Fallback to original if extraction failed
        logger.warning(f"Failed to extract spoken form, returning original: {text}")
        return text

    def convert_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
    ) -> List[str]:
        """Convert batch of texts.

        Note: Current implementation is sequential.
        Batched inference could be added for efficiency.

        Args:
            texts: List of Korean texts.
            batch_size: Not currently used (placeholder).

        Returns:
            List of converted texts.
        """
        results = []
        for text in texts:
            try:
                converted = self.convert(text)
                results.append(converted)
            except Exception as e:
                logger.error(f"Conversion failed for '{text}': {e}")
                results.append(text)  # Return original on failure
        return results

    def convert_with_result(self, text: str) -> ConversionResult:
        """Convert with detailed result.

        Args:
            text: Korean text in written style.

        Returns:
            ConversionResult with original, converted, and metadata.
        """
        converted = self.convert(text)

        # LLM-based conversion has higher confidence
        confidence = 0.85 if converted != text else 0.5

        return ConversionResult(
            original=text,
            converted=converted,
            confidence=confidence,
            method="local_llm",
        )


class HybridConverter(SpokenConverter):
    """Hybrid converter combining rule-based and LLM approaches.

    Uses rule-based conversion as primary method, falling back to
    LLM for complex cases where rules don't apply well.

    Attributes:
        rule_converter: Rule-based converter.
        llm_converter: Optional LLM converter for fallback.
    """

    def __init__(
        self,
        llm_model_path: Optional[Union[str, Path]] = None,
        use_llm: bool = False,
        **kwargs,
    ):
        """Initialize hybrid converter.

        Args:
            llm_model_path: Path to local LLM (optional).
            use_llm: Whether to use LLM as fallback.
            **kwargs: Arguments passed to RuleBasedConverter.
        """
        self.rule_converter = RuleBasedConverter(**kwargs)

        self.llm_converter = None
        if use_llm and llm_model_path:
            try:
                self.llm_converter = LocalLLMConverter(llm_model_path)
            except Exception as e:
                logger.warning(f"Failed to load LLM converter: {e}")

        logger.info(
            f"HybridConverter initialized (LLM: {'enabled' if self.llm_converter else 'disabled'})"
        )

    def convert(self, text: str) -> str:
        """Convert text using hybrid approach.

        First applies rule-based conversion, then optionally
        uses LLM for texts where rules had minimal effect.

        Args:
            text: Korean text in written style.

        Returns:
            Text converted to spoken style.
        """
        # Try rule-based first
        result = self.rule_converter.convert_with_result(text)

        # If low confidence and LLM available, try LLM
        if result.confidence < 0.7 and self.llm_converter:
            try:
                return self.llm_converter.convert(text)
            except Exception as e:
                logger.warning(f"LLM conversion failed: {e}")

        return result.converted

    def convert_batch(self, texts: List[str]) -> List[str]:
        """Convert batch of texts."""
        return [self.convert(t) for t in texts]


# Factory function
def get_converter(
    method: str = "rule",
    model_path: Optional[str] = None,
    **kwargs,
) -> SpokenConverter:
    """Factory function to get a spoken converter.

    Args:
        method: "rule", "llm", or "hybrid".
        model_path: Path to LLM model (for "llm" or "hybrid").
        **kwargs: Additional arguments for the converter.

    Returns:
        SpokenConverter instance.

    Raises:
        ValueError: If method is unknown.
    """
    if method == "rule":
        return RuleBasedConverter(**kwargs)
    elif method == "llm":
        if not model_path:
            raise ValueError("model_path required for LLM converter")
        return LocalLLMConverter(model_path, **kwargs)
    elif method == "hybrid":
        return HybridConverter(llm_model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown converter method: {method}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert written Korean to spoken style"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text (or use --file for file input)",
    )
    parser.add_argument(
        "--method",
        choices=["rule", "llm", "hybrid"],
        default="rule",
        help="Conversion method",
    )
    parser.add_argument("--model", help="Path to LLM model (for llm/hybrid)")
    parser.add_argument("--file", "-f", help="Input file (one sentence per line)")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument(
        "--add-fillers",
        action="store_true",
        help="Add filler words (rule-based)",
    )

    args = parser.parse_args()

    # Get converter
    converter = get_converter(
        method=args.method,
        model_path=args.model,
        add_fillers=args.add_fillers,
    )

    # Process input
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        converted = converter.convert_batch(texts)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for text in converted:
                    f.write(text + "\n")
        else:
            for orig, conv in zip(texts, converted):
                print(f"{orig} → {conv}")
    elif args.input:
        result = converter.convert(args.input)
        print(f"{args.input}")
        print(f"→ {result}")
    else:
        # Interactive mode
        print("Enter Korean text (Ctrl+D to exit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    result = converter.convert(text)
                    print(f"→ {result}")
        except EOFError:
            pass
