from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd


class GenericMissingnessStrategy(ABC):
    """Generic missingness strategy."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def remove(self, data: pd.Series) -> pd.Series:
        """Remove missingness."""
        pass


class RestoreMissingnessMixin(ABC):
    """Restore missingness mixin."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def restore(self, data: pd.Series) -> pd.Series:
        """Restore missingness."""
        pass


class DropMissingnessStrategy(GenericMissingnessStrategy):
    """Drop missingness strategy."""

    def __init__(self) -> None:
        super().__init__()

    def remove(self, data: pd.Series) -> pd.Series:
        """Drop missingness."""
        return data


class ImputeMissingnessStrategy(GenericMissingnessStrategy):
    """Impute missingness with mean strategy."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value.lower()

    def remove(self, data: pd.Series) -> pd.Series:
        """Impute missingness with mean."""
        if self.value == "mean":
            return data.fillna(data.mean())
        elif self.value == "median":
            return data.fillna(data.median())
        elif self.value == "mode":
            return data.fillna(data.mode()[0])
        else:
            return data.fillna(self.value)


class AugmentMissingnessStrategy(GenericMissingnessStrategy, RestoreMissingnessMixin):
    def __init__(self) -> None:
        super().__init__()

    def remove(
        self, data: Union[pd.Series, tuple[pd.Series, pd.DataFrame]], categorical: bool
    ) -> Union[pd.Series, pd.DataFrame]:
        """Impute missingness with model."""
        if categorical:
            if data.dtype.kind == "O":
                self.missing_value = data.unique()[0] + "_missing"
            else:
                self.missing_value = data.min() - 1
            return data.fillna(self.missing_value)
        else:
            original_data, transformed_data = data
            self.missing_column = original_data.name + "_missing"
            transformed_data = transformed_data.set_index(original_data.index[original_data.notnull()])
            transformed_data = transformed_data.reindex(original_data.index)
            transformed_data[original_data.name + "_missing"] = original_data.isnull().astype(int)
            transformed_data = transformed_data.fillna(0)
            return transformed_data

    def restore(self, data: Union[pd.Series, tuple[pd.Series, pd.DataFrame]], categorical: bool) -> pd.Series:
        """Restore missingness."""
        if categorical:
            # if the value of the data is missing, return np.nan
            return data.where(data == self.missing_value, np.nan)
        else:
            # if the value of the data is missing, return np.nan
            return data.where(data[self.missing_column] == 1, np.nan)
