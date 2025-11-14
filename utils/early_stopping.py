"""
Simple early stopping utility to halt training when validation loss
stagnates for a configurable number of epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EarlyStopping:
    """Tracks the best validation loss and decides when to stop training."""

    patience: int = 10
    min_delta: float = 0.0
    counter: int = field(default = 0, init = False)
    best_score: Optional[float] = field(default = None, init = False)
    should_stop: bool = field(default = False, init = False)

    def __call__(self, current_score: float) -> bool:
        """
        Update the early stopping state with a new validation loss.

        Returns:
            bool: True when training should stop.
        """
        # Initialize tracking during the first call.
        if self.best_score is None:
            self.best_score = current_score
            return False

        # Improvement is only registered when the loss decreases by min_delta.
        if current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0  # Reset patience because the model improved.
            return False

        # No improvement; increase counter and check if patience is exceeded.
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True  # Signal the caller to halt training.
        return self.should_stop

    def reset(self) -> None:
        """Clear the internal state to reuse the helper for a new training run."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
