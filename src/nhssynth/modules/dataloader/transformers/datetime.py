from typing import Optional

import numpy as np
import pandas as pd

from nhssynth.modules.dataloader.transformers.base import (
    ColumnTransformer,
    TransformerWrapper,
)


class DatetimeTransformer(TransformerWrapper):
    """
    A transformer to convert datetime features to numeric features. Before applying an underlying (wrapped) transformer.
    The datetime features are converted to nanoseconds since the epoch, and missing values are assigned to 0.0 under the `AugmentMissingnessStrategy`.

    Args:
        transformer: The [`ColumnTransformer`][nhssynth.modules.dataloader.transformers.base.ColumnTransformer] to wrap.

    After applying the transformer, the following attributes will be populated:

    Attributes:
        original_column_name: The name of the original column.
    """

    def __init__(self, transformer: ColumnTransformer) -> None:
        super().__init__(transformer)

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        """
        Firstly, the datetime data is floored to the nano-second level. Next, the floored data is converted to float nanoseconds since the epoch.
        The float value of `pd.NaT` under the operation above is then replaced with `np.nan` to ensure missing values are represented correctly.
        Finally, the wrapped transformer is applied to the data.

        Args:
            data: The column of data to transform.
            missingness_column: The column of missingness indicators to augment the data with.

        Returns:
            The transformed data.
        """
        self.original_column_name = data.name
        floored_data = pd.Series(data.dt.floor("ns").to_numpy().astype(float), name=data.name)
        nan_corrected_data = floored_data.replace(pd.to_datetime(pd.NaT).to_numpy().astype(float), np.nan)
        return super().apply(nan_corrected_data, missingness_column, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        The wrapped transformer's `revert` method is applied to the data. The data is then converted back to datetime format.

        Args:
            data: The full dataset including the column(s) to be reverted to their pre-transformer state.

        Returns:
            The reverted data.
        """
        reverted_data = super().revert(data, **kwargs)
        data[self.original_column_name] = pd.to_datetime(
            reverted_data[self.original_column_name].astype("Int64"), unit="ns"
        )
        return data
