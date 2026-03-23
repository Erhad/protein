"""Abstract base class for all MLDE methods."""

from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Interface shared by EVOLVEpro, BOES, ALDE, ftALDE.

    All methods operate on a fixed pool of variants with pre-computed
    embeddings and known (oracle) fitness values. The pool never changes —
    only the labeled/unlabeled split evolves each round.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate model on labeled set."""

    @abstractmethod
    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Return indices (into X_pool) of the next batch to label.
        Shape: (batch_size,)
        """
