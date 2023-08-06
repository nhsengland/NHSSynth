from typing import Optional

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.transformers.base import *


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
        self.original_column_name = data.name
        floored_data = pd.Series(data.dt.floor("ns").to_numpy().astype(float), name=data.name)
        nan_corrected_data = floored_data.replace(pd.to_datetime(pd.NaT).to_numpy().astype(float), np.nan)
        return super().apply(nan_corrected_data, missingness_column, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        reverted_data = super().revert(data, **kwargs)
        data[self.original_column_name] = pd.to_datetime(
            reverted_data[self.original_column_name].astype("Int64"), unit="ns"
        )
        return data
