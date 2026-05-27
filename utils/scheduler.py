import torch
import numpy as np
from config import CONFIG

class IncreaseRegOnPlateau:
    """Increase the regularization coefficient `reg` when a metric (accuracy) has stopped improving.

    Modeled directly after torch.optim.lr_scheduler.ReduceLROnPlateau but increases reg instead of decreasing LR.
    """
    def __init__(self, reg=0.1, mode='max', factor=1.5, patience=5,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_reg=0.05, max_reg=5.0, verbose=True):
        if factor <= 1.0:
            raise ValueError("Factor should be > 1.0 for increasing reg.")

        self.reg = float(reg)
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.min_reg = min_reg
        self.max_reg = max_reg
        self.verbose = verbose

        self.mode = mode
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.last_epoch = 0
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self.mode_worse = float('inf') if mode == "min" else float('-inf')
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        else:  # max abs
            return a > best + self.threshold

    def step(self, metrics):
        """Call this after every validation epoch with validation accuracy."""
        current = float(metrics)
        self.last_epoch += 1

        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            old_reg = self.reg
            new_reg = old_reg * self.factor
            new_reg = min(new_reg, self.max_reg)
            new_reg = max(new_reg, self.min_reg)

            if abs(new_reg - old_reg) > 1e-8:
                self.reg = new_reg
                if self.verbose:
                    print(f"IncreaseRegOnPlateau: reg increased from {old_reg} -> {self.reg} "
                          f"(epoch {self.last_epoch}, patience={self.patience})")

            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0