from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class ColumnTransformer(ABC):
    """A generic column transformer class to prototype all of the transformers applied via the [`MetaTransformer`][nhssynth.modules.dataloader.metatransformer.MetaTransformer]."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, data: pd.DataFrame, missingness_column: Optional[pd.Series]) -> None:
        """Apply the transformer to the data."""
        pass

    @abstractmethod
    def revert(self, data: pd.DataFrame) -> None:
        """Revert data to pre-transformer state."""
        pass


class TransformerWrapper(ABC):
    """
    A class to facilitate nesting of [`ColumnTransformer`s][nhssynth.modules.dataloader.transformers.base.ColumnTransformer].

    Args:
        wrapped_transformer: The [`ColumnTransformer`][nhssynth.modules.dataloader.transformers.base.ColumnTransformer] to wrap.
    """

    def __init__(self, wrapped_transformer: ColumnTransformer) -> None:
        super().__init__()
        self._wrapped_transformer = wrapped_transformer

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series], **kwargs) -> pd.DataFrame:
        """Clean method for applying the wrapped transformer to the data."""
        return self._wrapped_transformer.apply(data, missingness_column, **kwargs)

    def revert(self, data: pd.Series, **kwargs) -> pd.DataFrame:
        """Clean method for reverting the passed data via the wrapped transformer."""
        return self._wrapped_transformer.revert(data, **kwargs)
