from typing import Optional

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from sklearn.mixture import BayesianGaussianMixture


class ClusterTransformer(GenericTransformer):
    def __init__(
        self,
        n_components: int = 10,
        n_init: int = 5,
        init_params: str = "kmeans",
        random_state: int = 0,
        weight_threshold: float = 0.005,
    ) -> None:
        super().__init__()
        self._transformer = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            n_init=n_init,
            init_params=init_params,
            weight_concentration_prior=1e-3,
        )
        self._n_components = n_components
        self._weight_threshold = weight_threshold
        self._weights: Optional[list[float]] = None
        self._std_multiplier = 4
        self._missingness_column_name = None

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series] = None) -> pd.DataFrame:
        name = data.name
        if missingness_column is not None:
            self._missingness_column_name = missingness_column.name
            assert len(data[missingness_column == 1].unique()) == 1, "Only one missing value is supported"
            self._missingness_replacement_value = data[missingness_column == 1].unique()[0]
            missing_data = pd.DataFrame(data[missingness_column == 1].rename(f"{name}_normalised"))
            data = data[missingness_column == 0]
        index = data.index
        data = np.array(data.values.reshape(-1, 1), dtype=data.dtype.name.lower())

        # TODO consider whether we need to store min and max to clip on reversion
        self._transformer.fit(data)
        self.weights = self._transformer.weights_
        self.means = self._transformer.means_.reshape(-1)
        self.stds = np.sqrt(self._transformer.covariances_).reshape(-1)

        normalised_values = (data - self.means.reshape(1, -1)) / (self._std_multiplier * self.stds.reshape(1, -1))
        components = np.argmax(self._transformer.predict_proba(data), axis=1)
        normalised = normalised_values[np.arange(len(data)), components]
        normalised = np.clip(normalised, -1.0, 1.0)
        components = np.eye(self._n_components, dtype=int)[components]

        transformed_data = pd.DataFrame(
            np.hstack([normalised.reshape(-1, 1), components]),
            index=index,
            columns=[f"{name}_normalised"] + [f"{name}_c{i + 1}" for i in range(self._n_components)],
        )

        if missingness_column is not None:
            transformed_data = pd.concat(
                [pd.concat([transformed_data, missing_data]).sort_index().fillna(0.0), missingness_column], axis=1
            )

        transformed_data = transformed_data.astype({f"{name}_c{i + 1}": int for i in range(self._n_components)})
        self.columns = transformed_data.columns
        return transformed_data

    def revert(self, data: pd.DataFrame) -> pd.Series:
        assert not self.columns.empty and all(
            self.columns == data.columns
        ), "Input data columns do not match transformer columns"

        full_index = data.index
        if self._missingness_column_name is not None:
            data = data[data[self._missingness_column_name] == 0]
            data = data.drop(self._missingness_column_name, axis=1)
        index = data.index

        components = np.argmax(data.filter(regex=r".*_c\d+").values, axis=1)
        data = data.filter(like="_normalised").values.reshape(-1)
        data = np.clip(data, -1.0, 1.0)

        # recreate data
        mean_t = self.means[components]
        std_t = self.stds[components]
        return (
            pd.Series(data * self._std_multiplier * std_t + mean_t, index=index)
            .reindex(full_index)
            .fillna(self._missingness_replacement_value)
        )
