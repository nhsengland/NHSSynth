from abc import ABC, abstractmethod
from typing import Callable, Union

import pandas as pd
from nhssynth.modules.dataloader.missingness import GenericMissingnessStrategy


class GenericTransformer(ABC):
    """Generic Transformer class."""

    def __init__(self, missingness_strategy: Callable) -> None:
        super().__init__()
        self._missingness_strategy = missingness_strategy

    @abstractmethod
    def transform(self, data) -> None:
        """Transform data."""
        pass

    @abstractmethod
    def inverse_transform(self, data) -> None:
        """Inverse transform data."""
        pass

    def remove_missingness(self, data) -> None:
        """Tackle missingness in data."""
        self._missingness_strategy.remove(data)

    def restore_missingness(self, data) -> None:
        """Tackle missingness in data."""
        self._missingness_strategy.restore(data)


class TransformerWrapper(ABC):
    """Transformer Wrapper class."""

    def __init__(self, transformer: GenericTransformer) -> None:
        super().__init__()
        self._transformer = transformer

    @abstractmethod
    def transform(self, data) -> None:
        """Transform data."""
        pass

    @abstractmethod
    def inverse_transform(self, data) -> None:
        """Inverse transform data."""
        pass


class NullTransformer(GenericTransformer):
    """Null Transformer class."""

    def __init__(self, missingness_strategy: GenericMissingnessStrategy) -> None:
        super().__init__(missingness_strategy)

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data."""
        return data

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        """Inverse transform data."""
        return data
