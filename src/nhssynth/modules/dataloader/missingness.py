from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
import pandas as pd

# TODO fix circular import
if typing.TYPE_CHECKING:
    from nhssynth.modules.dataloader.metadata import ColumnMetaData


class GenericMissingnessStrategy(ABC):
    """Generic missingness strategy."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name

    @abstractmethod
    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """Remove missingness."""
        pass


class NullMissingnessStrategy(GenericMissingnessStrategy):
    """Null missingness strategy."""

    def __init__(self) -> None:
        super().__init__("none")

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """Do nothing."""
        return data


class DropMissingnessStrategy(GenericMissingnessStrategy):
    """Drop missingness strategy."""

    def __init__(self) -> None:
        super().__init__("drop")

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """Drop missingness in `column`"""
        return data.dropna(subset=[column_metadata.name]).reset_index(drop=True)


class ImputeMissingnessStrategy(GenericMissingnessStrategy):
    """Impute missingness with mean strategy."""

    def __init__(self, impute: Any) -> None:
        super().__init__("impute")
        self.impute = impute.lower() if isinstance(impute, str) else impute

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """Impute missingness."""
        if self.impute == "mean":
            self.imputation_value = data[column_metadata.name].mean()
        elif self.impute == "median":
            self.imputation_value = data[column_metadata.name].median()
        elif self.impute == "mode":
            self.imputation_value = data[column_metadata.name].mode()[0]
        else:
            self.imputation_value = self.impute
        data[column_metadata.name].fillna(self.imputation_value, inplace=True)
        return data


class AugmentMissingnessStrategy(GenericMissingnessStrategy):
    def __init__(self) -> None:
        super().__init__("augment")

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """Impute missingness with model."""
        if column_metadata.categorical:
            if column_metadata.dtype.kind == "O":
                self.missingness_carrier = column_metadata.name + "_missing"
            else:
                self.missingness_carrier = data[column_metadata.name].min() - 1
        else:
            self.missingness_carrier = column_metadata.name + "_missing"
            data[self.missingness_carrier] = data[column_metadata.name].isnull().astype(int)
        return data


MISSINGNESS_STRATEGIES: Final = {
    "none": NullMissingnessStrategy,
    "impute": ImputeMissingnessStrategy,
    "augment": AugmentMissingnessStrategy,
    "drop": DropMissingnessStrategy,
}
