from typing import Optional, Union

import pandas as pd
from nhssynth.modules.dataloader.missingness import GenericMissingnessStrategy
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from sklearn.preprocessing import OneHotEncoder


class OHETransformer(GenericTransformer):
    def __init__(self, drop: Optional[Union[list, str]] = None) -> None:
        super().__init__()
        self._drop = drop
        self._transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=self._drop)

    def apply(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            self._transformer.fit_transform(data.values.reshape(-1, 1)),
            columns=self._transformer.get_feature_names_out(input_features=[data.name]),
        )

    def revert(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self._transformer.inverse_transform(data.values).flatten(),
            index=data.index,
        )
