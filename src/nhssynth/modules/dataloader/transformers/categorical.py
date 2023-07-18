from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from sklearn.preprocessing import OneHotEncoder


class OHETransformer(GenericTransformer):
    def __init__(self, drop: Optional[Union[list, str]] = None) -> None:
        super().__init__()
        self._drop = drop
        self._transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=self._drop)
        self.missing_value = None

    def apply(self, data: pd.Series, missing_value: Optional[Any] = None) -> pd.DataFrame:
        self.original_column_name = data.name
        if missing_value:
            data = data.fillna(missing_value)
            self.missing_value = missing_value
        transformed_data = pd.DataFrame(
            self._transformer.fit_transform(data.values.reshape(-1, 1)),
            columns=self._transformer.get_feature_names_out(input_features=[data.name]),
        )
        self.new_column_names = transformed_data.columns
        return transformed_data

    def revert(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.original_column_name] = pd.Series(
            self._transformer.inverse_transform(data[self.new_column_names].values).flatten(),
            index=data.index,
            name=self.original_column_name,
        )
        if self.missing_value:
            data[self.original_column_name] = data[self.original_column_name].replace(self.missing_value, np.nan)
        data.drop(self.new_column_names, axis=1, inplace=True)
        return data
