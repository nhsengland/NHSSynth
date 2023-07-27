from typing import Optional

import pandas as pd
from nhssynth.modules.dataloader.transformers.base import *


class DatetimeTransformer(TransformerWrapper):
    """
    A transformer to convert datetime features to numeric features. Before applying an underlying (wrapped) transformer.
    The datetime features are converted to nanoseconds since the epoch, and missing values are assigned to 0.0 under the `AugmentMissingnessStrategy`.

    Args:
        transformer: The [`ColumnTransformer`][nhssynth.modules.dataloader.transformers.base.ColumnTransformer] to wrap.
        format: The string format of the datetime feature, CURRENTLY UNUSED.

    After applying the transformer, the following attributes will be populated:

    Attributes:
        original_column_name: The name of the original column.
    """

    def __init__(self, transformer: ColumnTransformer, format: Optional[str] = None) -> None:
        super().__init__(transformer)
        self._format = format

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        self.original_column_name = data.name
        return super().apply(data, missingness_column, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        reverted_data = super().revert(data, **kwargs)
        data[self.original_column_name] = pd.to_datetime(
            reverted_data[self.original_column_name].astype("Int64"), unit="ns"
        )
        return data
