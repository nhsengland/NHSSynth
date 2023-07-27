import warnings
from typing import Optional

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.transformers.base import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture


class ClusterContinuousTransformer(ColumnTransformer):
    """
    A transformer to cluster continuous features via sklearn's `BayesianGaussianMixture`.
    Essentially wraps the process of fitting the BGM model and generating cluster assignments and normalised values for the data to comply with the `ColumnTransformer` interface.

    Args:
        n_components: The number of components to use in the BGM model.
        n_init: The number of initialisations to use in the BGM model.
        init_params: The initialisation method to use in the BGM model.
        random_state: The random state to use in the BGM model.
        max_iter: The maximum number of iterations to use in the BGM model.

    After applying the transformer, the following attributes will be populated:

    Attributes:
        weights: The weights of the components in the BGM model.
        means: The means of the components in the BGM model.
        stds: The standard deviations of the components in the BGM model.
        new_column_names: The names of the columns generated by the transformer (one for the normalised values and one for each cluster component).
    """

    def __init__(
        self,
        n_components: int = 10,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: int = 0,
        max_iter: int = 1000,
    ) -> None:
        super().__init__()
        self._transformer = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            n_init=n_init,
            init_params=init_params,
            max_iter=max_iter,
            weight_concentration_prior=1e-3,
        )
        self._n_components = n_components
        # self._weights: Optional[list[float]] = None
        self._std_multiplier = 4
        self._missingness_column_name = None
        self._max_iter = max_iter

    def apply(self, data: pd.Series, missingness_column: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply the transformer to the data via sklearn's `BayesianGaussianMixture`'s `fit` and `predict_proba` methods.
        Name the new columns via the original column name.

        If `missingness_column` is provided, use this to extract the non-missing data; the missing values are assigned to a new pseudo-cluster with mean 0
        (i.e. all values in the normalised column are 0.0).

        Args:
            data: The column of data to transform.
            missingness_column: The column of data representing missingness, this is only used as part of the `AugmentMissingnessStrategy`.

        Returns:
            The transformed data (will be multiple columns if `n_components` > 1 at initialisation).
        """
        self.original_column_name = data.name
        if missingness_column is not None:
            self._missingness_column_name = missingness_column.name
            missing_data = pd.DataFrame(data[missingness_column == 1].rename(f"{self.original_column_name}_normalised"))
            data = data[missingness_column == 0]
        index = data.index
        data = np.array(data.values.reshape(-1, 1), dtype=data.dtype.name.lower())

        # TODO consider whether we need to store min and max to clip on reversion
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
            columns=[f"{self.original_column_name}_normalised"]
            + [f"{self.original_column_name}_c{i + 1}" for i in range(self._n_components)],
        )

        if missingness_column is not None:
            transformed_data = pd.concat(
                [pd.concat([transformed_data, missing_data]).sort_index().fillna(0.0), missingness_column], axis=1
            )

        self.new_column_names = transformed_data.columns
        return transformed_data.astype(
            {f"{self.original_column_name}_c{i + 1}": int for i in range(self._n_components)}
        )

    def revert(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Revert data to pre-transformer state via the means and stds of the BGM. Extract the relevant columns from the data via the `new_column_names` attribute.
        If `missingness_column` was provided to the `apply` method, drop the missing values from the data before reverting and use the `full_index` to
        reintroduce missing values when `original_column_name` is constructed.

        Args:
            data: The full dataset including the column(s) to be reverted to their pre-transformer state.

        Returns:
            The dataset with a single continuous column that is analogous to the original column, with the same name, and without the generated format columns.
        """
        working_data = data[self.new_column_names]
        full_index = working_data.index
        if self._missingness_column_name is not None:
            working_data = working_data[working_data[self._missingness_column_name] == 0]
            working_data = working_data.drop(self._missingness_column_name, axis=1)
        index = working_data.index

        components = np.argmax(working_data.filter(regex=r".*_c\d+").values, axis=1)
        working_data = working_data.filter(like="_normalised").values.reshape(-1)
        working_data = np.clip(working_data, -1.0, 1.0)

        mean_t = self.means[components]
        std_t = self.stds[components]
        data[self.original_column_name] = pd.Series(
            working_data * self._std_multiplier * std_t + mean_t, index=index, name=self.original_column_name
        ).reindex(full_index)
        data.drop(self.new_column_names, axis=1, inplace=True)
        return data
