import contextlib
import dataclasses
import logging
import time
from typing import Protocol

import torch

logger = logging.getLogger("utils")


@dataclasses.dataclass
class TrainState:
    """
    Training state that tracks progress and metrics.

    This class is serializable for checkpoint save/load operations.
    All fields needed for resume are included.
    """
    max_steps: int
    step: int = 0
    elapsed_time: float = 0.0
    n_seen_tokens: int = 0
    this_step_time: float = 0.0
    begin_step_time: float = 0.0
    this_step_tokens: int = 0  # Added explicit field

    # Current step metrics
    this_eval_perplexity: float | None = None
    this_eval_loss: float | None = None
    this_audio_loss: float | None = None
    this_text_loss: float | None = None
    this_train_loss: float | None = None  # Added for metric tracking

    # User stream losses (Full Duplex mode with dep_q=16)
    this_moshi_audio_loss: float | None = None
    this_user_audio_loss: float | None = None

    # Best model tracking (for checkpoint management)
    best_metric: float | None = None
    best_step: int = 0
    metric_type: str = "eval_loss"  # Type of metric being tracked

    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens

        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += self.this_step_tokens

        self.begin_step_time = time.time()

    def get_metric(self, metric_type: str | None = None) -> float | None:
        """
        Get the current value of the specified metric.

        Args:
            metric_type: One of "train_loss", "eval_loss", "eval_perplexity"
                        If None, uses self.metric_type

        Returns:
            The metric value, or None if not available
        """
        metric_type = metric_type or self.metric_type

        if metric_type == "train_loss":
            return self.this_train_loss
        elif metric_type == "eval_loss":
            return self.this_eval_loss
        elif metric_type == "eval_perplexity":
            return self.this_eval_perplexity
        else:
            logger.warning(f"Unknown metric_type: {metric_type}")
            return None

    def update_best(self, metric_value: float, metric_best: str = "min") -> bool:
        """
        Update best metric if current value is better.

        Args:
            metric_value: Current metric value
            metric_best: "min" (lower is better) or "max" (higher is better)

        Returns:
            True if this is a new best, False otherwise
        """
        if self.best_metric is None:
            self.best_metric = metric_value
            self.best_step = self.step
            return True

        is_better = (
            (metric_best == "min" and metric_value < self.best_metric) or
            (metric_best == "max" and metric_value > self.best_metric)
        )

        if is_better:
            self.best_metric = metric_value
            self.best_step = self.step
            return True

        return False

    def to_dict(self) -> dict:
        """
        Serialize TrainState to dictionary for checkpoint saving.

        Returns:
            Dictionary containing all serializable state
        """
        return {
            "max_steps": self.max_steps,
            "step": self.step,
            "elapsed_time": self.elapsed_time,
            "n_seen_tokens": self.n_seen_tokens,
            "this_step_tokens": self.this_step_tokens,
            "this_eval_perplexity": self.this_eval_perplexity,
            "this_eval_loss": self.this_eval_loss,
            "this_audio_loss": self.this_audio_loss,
            "this_text_loss": self.this_text_loss,
            "this_train_loss": self.this_train_loss,
            "this_moshi_audio_loss": self.this_moshi_audio_loss,
            "this_user_audio_loss": self.this_user_audio_loss,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "metric_type": self.metric_type,
        }

    @classmethod
    def from_dict(cls, data: dict, max_steps: int | None = None) -> "TrainState":
        """
        Deserialize TrainState from dictionary for checkpoint loading.

        Args:
            data: Dictionary from to_dict()
            max_steps: Override max_steps (useful if config changed)

        Returns:
            Restored TrainState instance
        """
        # Use provided max_steps or fall back to saved value
        effective_max_steps = max_steps if max_steps is not None else data.get("max_steps", 0)

        state = cls(max_steps=effective_max_steps)

        # Restore core progress
        state.step = data.get("step", 0)
        state.elapsed_time = data.get("elapsed_time", 0.0)
        state.n_seen_tokens = data.get("n_seen_tokens", 0)
        state.this_step_tokens = data.get("this_step_tokens", 0)

        # Restore metrics
        state.this_eval_perplexity = data.get("this_eval_perplexity")
        state.this_eval_loss = data.get("this_eval_loss")
        state.this_audio_loss = data.get("this_audio_loss")
        state.this_text_loss = data.get("this_text_loss")
        state.this_train_loss = data.get("this_train_loss")
        state.this_moshi_audio_loss = data.get("this_moshi_audio_loss")
        state.this_user_audio_loss = data.get("this_user_audio_loss")

        # Restore best tracking
        state.best_metric = data.get("best_metric")
        state.best_step = data.get("best_step", 0)
        state.metric_type = data.get("metric_type", "eval_loss")

        return state

    @property
    def wps(self):
        """Words per second for current step."""
        if self.this_step_time > 0:
            return self.this_step_tokens / self.this_step_time
        return 0.0

    @property
    def avg_wps(self):
        """Average words per second over all steps."""
        if self.elapsed_time > 0:
            return self.n_seen_tokens / self.elapsed_time
        return 0.0

    @property
    def eta(self):
        """Estimated time remaining in seconds."""
        if self.step > 0:
            steps_left = self.max_steps - self.step
            avg_time_per_step = self.elapsed_time / self.step
            return steps_left * avg_time_per_step
        return 0.0

    def __repr__(self) -> str:
        best_str = f"{self.best_metric:.4f}@{self.best_step}" if self.best_metric is not None else "N/A"
        return (
            f"TrainState(step={self.step}/{self.max_steps}, "
            f"elapsed={self.elapsed_time:.1f}s, "
            f"tokens={self.n_seen_tokens}, "
            f"best={best_str})"
        )


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    """
    Logging the closing to be sure something is not hanging at exit time
    """
    try:
        setattr(thing, "wrapped_by_closing", True)
        yield
    finally:
        logger.info(f"Closing: {name}")
        try:
            thing.close()
        except Exception:
            logger.error(f"Error while closing {name}!")
            raise
        logger.info(f"Closed: {name}")
