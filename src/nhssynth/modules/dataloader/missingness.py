from __future__ import annotations

import typing
import warnings
from abc import ABC, abstractmethod
from typing import Any, Final

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
        """
        Drop rows containing missing values in the appropriate column.

        Args:
            data: The dataset.
            column_metadata: The column metadata.

        Returns:
            The dataset with rows containing missing values in the appropriate column dropped.
        """
        return data.dropna(subset=[column_metadata.name]).reset_index(drop=True)


class ImputeMissingnessStrategy(GenericMissingnessStrategy):
    """Impute missingness with mean strategy."""

    def __init__(self, impute: Any) -> None:
        super().__init__("impute")
        self.impute = impute.lower() if isinstance(impute, str) else impute

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """
        Impute missingness in the data via the `impute` strategy. 'Special' values trigger specific behaviour.

        Args:
            data: The dataset.
            column_metadata: The column metadata.

        Returns:
            The dataset with missing values in the appropriate column replaced with imputed ones.
        """
        if (self.impute == "mean" or self.impute == "median") and column_metadata.categorical:
            warnings.warn("Cannot impute mean or median for categorical data, using mode instead.")
            self.imputation_value = data[column_metadata.name].mode()[0]
        elif self.impute == "mean":
            self.imputation_value = data[column_metadata.name].mean()
        elif self.impute == "median":
            self.imputation_value = data[column_metadata.name].median()
        elif self.impute == "mode":
            self.imputation_value = data[column_metadata.name].mode()[0]
        else:
            self.imputation_value = self.impute
        self.imputation_value = column_metadata.dtype.type(self.imputation_value)
        try:
            data[column_metadata.name] = data[column_metadata.name].fillna(self.imputation_value)
        except AssertionError:
            raise ValueError(f"Could not impute '{self.imputation_value}' into column: '{column_metadata.name}'.")
        return data


class AugmentMissingnessStrategy(GenericMissingnessStrategy):
    def __init__(self) -> None:
        super().__init__("augment")

    def remove(self, data: pd.DataFrame, column_metadata: ColumnMetaData) -> pd.DataFrame:
        """
        Impute missingness with the model. To do this we create a new column for continuous features and a new category for categorical features.

        Args:
            data: The dataset.
            column_metadata: The column metadata enabling the correct set up of the missingness strategy.

        Returns:
            The dataset, potentially with a new column representing the missingness for the column added.
        """
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
    # "none": NullMissingnessStrategy,
    "impute": ImputeMissingnessStrategy,
    "augment": AugmentMissingnessStrategy,
    "drop": DropMissingnessStrategy,
}
