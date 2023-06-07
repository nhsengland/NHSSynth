from typing import Optional

import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import *


class DatetimeTransformer(TransformerWrapper):
    def __init__(self, transformer: GenericTransformer, format: Optional[str] = None, utc: bool = False) -> None:
        super().__init__(transformer)
        self._format = format
        self._utc = utc

    def transform(self, data: pd.Series) -> pd.DataFrame:
        pd.to_numeric(data).to_numpy().astype(float)
        return self._transformer.transform(data)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        data = self._transformer.inverse_transform(data)
        return pd.to_datetime(data, format=self._format, utc=self._utc)
