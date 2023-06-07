from typing import Callable, Optional

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from sklearn.mixture import BayesianGaussianMixture


class ClusterTransformer(GenericTransformer):
    def __init__(
        self,
        missingness_strategy: Callable[[pd.Series], pd.Series],
        n_components: int = 10,
        n_init: int = 5,
        init_params: str = "kmeans",
        random_state: int = 0,
        weight_threshold: float = 0.005,
    ) -> None:
        super().__init__(missingness_strategy)
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

    def transform(self, data: pd.Series) -> pd.DataFrame:
        # Fit model to data
        name = data.name
        data = data.dropna()
        data = data.values.reshape(-1, 1)

        self.model.fit(data)

        self.weights = self.model.weights_
        means = self.model.means_.reshape(1, self.n_components)

        # Get weights and means of components
        self.weights = self.model.weights_
        means = self.model.means_

        # Normalize data
        normalized_data = np.zeros_like(data.values)
        for i in range(self.n_components):
            mask = self.model.predict(data.values.reshape(-1, 1)) == i
            normalized_data[mask] = (data.values[mask] - means[i]) / (self.std_multiplier * self.model.covariances_[i])

        # Clip normalized data to [-1, 1]
        normalized_data = np.clip(normalized_data, -1, 1)

        # Return normalized data as DataFrame
        return pd.DataFrame(
            normalized_data.reshape(-1, 1),
            columns=[],
        )

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        # Get original data
        original_data = np.zeros_like(data.values)
        for i in range(self.n_components):
            mask = self.model.predict(original_data.reshape(-1, 1)) == i
            original_data[mask] = (
                data.values[mask] * (self.std_multiplier * self.model.covariances_[i]) + self.model.means_[i]
            )

        # Return original data as Series
        return pd.Series(
            original_data.flatten(),
            index=data.index,
        )
