# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""English Data Mixing for K-Moshi Bilingual Capability.

This module provides functionality to mix English dialogue data
into the Korean corpus to enable:
1. Bilingual capability (respond to English queries)
2. Code-switching support (Korean with English terms)
3. Prevent catastrophic forgetting of original Moshi's English

Supported English Corpora:
- DailyDialog: 13K daily life conversations
- EmpatheticDialogues: 25K empathetic conversations
- PersonaChat: 10K persona-based conversations

Usage:
    mixer = EnglishMixer(english_ratio=0.1, code_switch_ratio=0.05)
    mixed = mixer.mix_corpus(korean_dialogues, english_dialogues)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single turn in a dialogue.

    Attributes:
        speaker: Speaker identifier ("SPEAKER_MAIN" or "SPEAKER_USER").
        text: The utterance text.
        language: Language code ("ko", "en", or "ko+en" for mixed).
    """
    speaker: str
    text: str
    language: str = "ko"


@dataclass
class MixedDialogue:
    """A dialogue with potentially mixed languages.

    Attributes:
        turns: List of dialogue turns.
        source: Source dataset identifier.
        language_ratio: Tuple of (korean_ratio, english_ratio).
        metadata: Optional additional metadata.
    """
    turns: List[DialogueTurn]
    source: str
    language_ratio: Tuple[float, float] = (1.0, 0.0)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turns": [
                {"speaker": t.speaker, "text": t.text, "language": t.language}
                for t in self.turns
            ],
            "source": self.source,
            "language_ratio": list(self.language_ratio),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MixedDialogue":
        """Create from dictionary."""
        turns = [
            DialogueTurn(
                speaker=t["speaker"],
                text=t["text"],
                language=t.get("language", "ko"),
            )
            for t in data["turns"]
        ]
        return cls(
            turns=turns,
            source=data.get("source", "unknown"),
            language_ratio=tuple(data.get("language_ratio", [1.0, 0.0])),
            metadata=data.get("metadata", {}),
        )


class EnglishMixer:
    """Mix English dialogues into Korean corpus for bilingual capability.

    This class provides methods to:
    1. Load and convert English dialogue corpora
    2. Mix English dialogues into Korean corpus
    3. Inject English terms into Korean text (code-switching)

    Attributes:
        english_ratio: Ratio of pure English dialogues (0.0-1.0).
        code_switch_ratio: Ratio of Korean dialogues with English terms.
        seed: Random seed for reproducibility.
    """

    # Technical terms commonly used in Korean conversations
    TECH_TERMS_KO_EN = {
        "인공지능": "AI",
        "응용프로그램": "application",
        "애플리케이션": "app",
        "서버": "server",
        "데이터베이스": "database",
        "사용자 인터페이스": "UI",
        "사용자 경험": "UX",
        "프로그래밍": "programming",
        "알고리즘": "algorithm",
        "기계학습": "machine learning",
        "딥러닝": "deep learning",
        "클라우드": "cloud",
        "인터넷": "internet",
        "웹사이트": "website",
        "소프트웨어": "software",
        "하드웨어": "hardware",
    }

    def __init__(
        self,
        english_ratio: float = 0.1,
        code_switch_ratio: float = 0.05,
        seed: Optional[int] = None,
    ):
        """Initialize English mixer.

        Args:
            english_ratio: Ratio of pure English dialogues (0.0-1.0).
                Default 0.1 (10% English).
            code_switch_ratio: Ratio of Korean dialogues with English terms.
                Default 0.05 (5% code-switched).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If ratios are out of valid range.
        """
        if not 0.0 <= english_ratio <= 1.0:
            raise ValueError(f"english_ratio must be 0.0-1.0, got {english_ratio}")
        if not 0.0 <= code_switch_ratio <= 1.0:
            raise ValueError(
                f"code_switch_ratio must be 0.0-1.0, got {code_switch_ratio}"
            )
        if english_ratio + code_switch_ratio > 1.0:
            raise ValueError(
                f"Sum of ratios ({english_ratio + code_switch_ratio}) exceeds 1.0"
            )

        self.english_ratio = english_ratio
        self.code_switch_ratio = code_switch_ratio
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        logger.info(
            f"EnglishMixer initialized: english={english_ratio:.0%}, "
            f"code_switch={code_switch_ratio:.0%}"
        )

    def load_english_corpus(
        self,
        corpus_path: Union[str, Path],
        corpus_type: str = "dailydialog",
    ) -> List[dict]:
        """Load English dialogue corpus.

        Args:
            corpus_path: Path to corpus file or directory.
            corpus_type: Type of corpus ("dailydialog", "empathetic", "persona").

        Returns:
            List of dialogue dictionaries.

        Raises:
            FileNotFoundError: If corpus path doesn't exist.
            ValueError: If corpus type is unknown.
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        if corpus_type == "dailydialog":
            return self._load_dailydialog(corpus_path)
        elif corpus_type == "empathetic":
            return self._load_empathetic(corpus_path)
        elif corpus_type == "persona":
            return self._load_persona(corpus_path)
        elif corpus_type == "jsonl":
            return self._load_jsonl(corpus_path)
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")

    def _load_dailydialog(self, path: Path) -> List[dict]:
        """Load DailyDialog corpus."""
        dialogues = []

        # DailyDialog format: one dialogue per line, utterances separated by __eou__
        dialog_file = path / "dialogues_text.txt" if path.is_dir() else path

        with open(dialog_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
                if len(utterances) >= 2:
                    dialogues.append({
                        "turns": utterances,
                        "source": f"dailydialog_{line_num}",
                    })

        logger.info(f"Loaded {len(dialogues)} DailyDialog conversations")
        return dialogues

    def _load_empathetic(self, path: Path) -> List[dict]:
        """Load EmpatheticDialogues corpus."""
        dialogues = []

        # EmpatheticDialogues format: CSV with context and utterances
        csv_file = path / "train.csv" if path.is_dir() else path

        try:
            import csv
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                current_conv_id = None
                current_turns = []

                for row in reader:
                    conv_id = row.get("conv_id")
                    utterance = row.get("utterance", "")

                    if conv_id != current_conv_id:
                        if current_turns and len(current_turns) >= 2:
                            dialogues.append({
                                "turns": current_turns,
                                "source": f"empathetic_{current_conv_id}",
                            })
                        current_conv_id = conv_id
                        current_turns = []

                    if utterance:
                        current_turns.append(utterance)

                # Don't forget the last conversation
                if current_turns and len(current_turns) >= 2:
                    dialogues.append({
                        "turns": current_turns,
                        "source": f"empathetic_{current_conv_id}",
                    })
        except Exception as e:
            logger.error(f"Error loading EmpatheticDialogues: {e}")

        logger.info(f"Loaded {len(dialogues)} EmpatheticDialogues conversations")
        return dialogues

    def _load_persona(self, path: Path) -> List[dict]:
        """Load PersonaChat corpus."""
        dialogues = []

        # PersonaChat format: JSON with personas and dialogues
        json_file = path / "train_self_original.json" if path.is_dir() else path

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

            for entry in data:
                utterances = entry.get("utterances", [])
                if utterances and len(utterances) >= 2:
                    dialogues.append({
                        "turns": utterances,
                        "source": "persona",
                    })

        logger.info(f"Loaded {len(dialogues)} PersonaChat conversations")
        return dialogues

    def _load_jsonl(self, path: Path) -> List[dict]:
        """Load generic JSONL corpus."""
        dialogues = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "turns" in data:
                    dialogues.append(data)

        logger.info(f"Loaded {len(dialogues)} dialogues from JSONL")
        return dialogues

    def convert_to_moshi_format(
        self,
        english_dialogue: dict,
        swap_speakers: bool = False,
    ) -> MixedDialogue:
        """Convert English dialogue to K-Moshi format.

        Args:
            english_dialogue: English dialogue dictionary with "turns" key.
            swap_speakers: If True, swap SPEAKER_MAIN and SPEAKER_USER.

        Returns:
            MixedDialogue in K-Moshi format.
        """
        turns = []
        speakers = ["SPEAKER_MAIN", "SPEAKER_USER"]
        if swap_speakers:
            speakers = speakers[::-1]

        utterances = english_dialogue.get("turns", [])
        for i, utterance in enumerate(utterances):
            speaker = speakers[i % 2]
            turns.append(DialogueTurn(
                speaker=speaker,
                text=utterance,
                language="en",
            ))

        return MixedDialogue(
            turns=turns,
            source=english_dialogue.get("source", "english"),
            language_ratio=(0.0, 1.0),
            metadata={"original_language": "en"},
        )

    def inject_english_terms(
        self,
        korean_text: str,
        injection_prob: float = 0.3,
    ) -> str:
        """Inject English terms into Korean text for code-switching.

        This simulates natural code-switching where Korean speakers
        use English technical terms.

        Args:
            korean_text: Original Korean text.
            injection_prob: Probability of replacing each term.

        Returns:
            Korean text with some English terms.
        """
        result = korean_text

        for korean, english in self.TECH_TERMS_KO_EN.items():
            if korean in result and random.random() < injection_prob:
                result = result.replace(korean, english, 1)

        return result

    def mix_corpus(
        self,
        korean_dialogues: List[dict],
        english_dialogues: List[dict],
    ) -> List[MixedDialogue]:
        """Mix Korean and English dialogues according to configured ratios.

        Args:
            korean_dialogues: List of Korean dialogue dictionaries.
            english_dialogues: List of English dialogue dictionaries.

        Returns:
            List of MixedDialogue objects with specified language ratios.
        """
        total = len(korean_dialogues)
        n_english = int(total * self.english_ratio)
        n_code_switch = int(total * self.code_switch_ratio)
        n_korean = total - n_english - n_code_switch

        logger.info(
            f"Mixing corpus: {n_korean} Korean, {n_english} English, "
            f"{n_code_switch} code-switched"
        )

        mixed = []

        # Pure Korean dialogues
        for dialogue in korean_dialogues[:n_korean]:
            mixed.append(self._convert_korean_dialogue(dialogue))

        # Pure English dialogues
        sampled_english = random.sample(
            english_dialogues,
            min(n_english, len(english_dialogues)),
        )
        for dialogue in sampled_english:
            # Randomly swap speakers for variety
            swap = random.random() < 0.5
            mixed.append(self.convert_to_moshi_format(dialogue, swap_speakers=swap))

        # Code-switched Korean dialogues
        for dialogue in korean_dialogues[n_korean:n_korean + n_code_switch]:
            mixed.append(self._convert_with_code_switch(dialogue))

        # Shuffle to distribute evenly
        random.shuffle(mixed)

        logger.info(f"Created mixed corpus with {len(mixed)} dialogues")
        return mixed

    def _convert_korean_dialogue(self, dialogue: dict) -> MixedDialogue:
        """Convert pure Korean dialogue to MixedDialogue format."""
        turns = []
        for turn in dialogue.get("turns", []):
            if isinstance(turn, dict):
                turns.append(DialogueTurn(
                    speaker=turn.get("speaker", "SPEAKER_MAIN"),
                    text=turn.get("text", ""),
                    language="ko",
                ))
            elif isinstance(turn, str):
                # Simple string format - alternate speakers
                speaker = "SPEAKER_MAIN" if len(turns) % 2 == 0 else "SPEAKER_USER"
                turns.append(DialogueTurn(
                    speaker=speaker,
                    text=turn,
                    language="ko",
                ))

        return MixedDialogue(
            turns=turns,
            source=dialogue.get("source", "korean"),
            language_ratio=(1.0, 0.0),
        )

    def _convert_with_code_switch(self, dialogue: dict) -> MixedDialogue:
        """Convert Korean dialogue with English code-switching."""
        turns = []
        for turn in dialogue.get("turns", []):
            if isinstance(turn, dict):
                text = self.inject_english_terms(turn.get("text", ""))
                turns.append(DialogueTurn(
                    speaker=turn.get("speaker", "SPEAKER_MAIN"),
                    text=text,
                    language="ko+en",
                ))
            elif isinstance(turn, str):
                text = self.inject_english_terms(turn)
                speaker = "SPEAKER_MAIN" if len(turns) % 2 == 0 else "SPEAKER_USER"
                turns.append(DialogueTurn(
                    speaker=speaker,
                    text=text,
                    language="ko+en",
                ))

        return MixedDialogue(
            turns=turns,
            source=dialogue.get("source", "korean_mixed"),
            language_ratio=(0.9, 0.1),  # Approximate
        )

    def save_mixed_corpus(
        self,
        dialogues: List[MixedDialogue],
        output_path: Union[str, Path],
    ):
        """Save mixed corpus to JSONL file.

        Args:
            dialogues: List of MixedDialogue objects.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for dialogue in dialogues:
                f.write(json.dumps(dialogue.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved mixed corpus to {output_path}")

    def get_statistics(
        self,
        dialogues: List[MixedDialogue],
    ) -> dict:
        """Get statistics about the mixed corpus.

        Args:
            dialogues: List of MixedDialogue objects.

        Returns:
            Dictionary with corpus statistics.
        """
        stats = {
            "total_dialogues": len(dialogues),
            "korean_only": 0,
            "english_only": 0,
            "code_switched": 0,
            "total_turns": 0,
            "avg_turns_per_dialogue": 0.0,
        }

        for dialogue in dialogues:
            ko_ratio, en_ratio = dialogue.language_ratio
            if en_ratio == 0:
                stats["korean_only"] += 1
            elif ko_ratio == 0:
                stats["english_only"] += 1
            else:
                stats["code_switched"] += 1

            stats["total_turns"] += len(dialogue.turns)

        if dialogues:
            stats["avg_turns_per_dialogue"] = stats["total_turns"] / len(dialogues)

        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mix English into Korean corpus")
    parser.add_argument("korean_corpus", help="Korean corpus JSONL file")
    parser.add_argument("english_corpus", help="English corpus file/directory")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument(
        "--english-ratio",
        type=float,
        default=0.1,
        help="Ratio of pure English (default: 0.1)",
    )
    parser.add_argument(
        "--code-switch-ratio",
        type=float,
        default=0.05,
        help="Ratio of code-switched (default: 0.05)",
    )
    parser.add_argument(
        "--corpus-type",
        choices=["dailydialog", "empathetic", "persona", "jsonl"],
        default="dailydialog",
        help="English corpus type",
    )
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Initialize mixer
    mixer = EnglishMixer(
        english_ratio=args.english_ratio,
        code_switch_ratio=args.code_switch_ratio,
        seed=args.seed,
    )

    # Load corpora
    with open(args.korean_corpus, encoding="utf-8") as f:
        korean = [json.loads(line) for line in f]

    english = mixer.load_english_corpus(args.english_corpus, args.corpus_type)

    # Mix and save
    mixed = mixer.mix_corpus(korean, english)
    mixer.save_mixed_corpus(mixed, args.output)

    # Print statistics
    stats = mixer.get_statistics(mixed)
    print("\nCorpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
