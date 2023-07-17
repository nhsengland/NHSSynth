from typing import Optional

import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import *


class DatetimeTransformer(TransformerWrapper):
    def __init__(self, transformer: GenericTransformer, format: Optional[str] = None) -> None:
        super().__init__(transformer)
        self._format = format

    def apply(self, data: pd.Series, *args, **kwargs) -> pd.DataFrame:
        return super().apply(pd.Series(pd.to_numeric(data).to_numpy().astype(float), name=data.name), *args, **kwargs)

    def revert(self, data: pd.Series, *args, **kwargs) -> pd.Series:
        return pd.to_datetime(super().revert(data, *args, **kwargs), format=self._format)
