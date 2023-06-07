from typing import Optional, Union

import pandas as pd
from nhssynth.modules.dataloader.missingness import (
    AugmentMissingnessStrategy,
    GenericMissingnessStrategy,
)
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from sklearn.preprocessing import OneHotEncoder


class OHETransformer(GenericTransformer):
    def __init__(
        self, missingness_strategy: GenericMissingnessStrategy, drop: Optional[Union[list, str]] = None
    ) -> None:
        super().__init__(missingness_strategy)
        self._drop = drop
        self._transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=self._drop)

    def transform(self, data: pd.Series) -> pd.DataFrame:
        if isinstance(self._missingness_strategy, AugmentMissingnessStrategy):
            data = self.remove_missingness(data)
        return pd.DataFrame(
            self._transformer.fit_transform(data.values.reshape(-1, 1)),
            columns=self._transformer.get_feature_names_out(),
        )

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        data = pd.Series(
            self._transformer.inverse_transform(data.values).flatten(),
            index=data.index,
        )
        if isinstance(self._missingness_strategy, AugmentMissingnessStrategy):
            data = self.remove_missingness(data)
        return data
