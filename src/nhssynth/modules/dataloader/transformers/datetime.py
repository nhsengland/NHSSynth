from typing import Optional

import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import *


class DatetimeTransformer(TransformerWrapper):
    def __init__(self, transformer: GenericTransformer, format: Optional[str] = None) -> None:
        super().__init__(transformer)
        self._format = format

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        self.column_name = data.name
        numeric_data = pd.Series(data.dt.floor("ns").to_numpy().astype(float), name=data.name)
        if missingness_column is not None:
            numeric_data[missingness_column == 1] = 0.0
        return super().apply(numeric_data, missingness_column, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        reverted_data = super().revert(data, **kwargs)
        data[self.column_name] = pd.to_datetime(reverted_data[self.column_name].astype("Int64"), unit="ns")
        return data
