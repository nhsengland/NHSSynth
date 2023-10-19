import pathlib
import sys
from typing import Any, Optional, Self, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import MISSINGNESS_STRATEGIES


class MetaTransformer:
    """
    The metatransformer is responsible for transforming input dataset into a format that can be used by the `model` module, and for transforming
    this module's output back to the original format of the input dataset.

    Args:
        dataset: The raw input DataFrame.
        metadata: Optionally, a [`MetaData`][nhssynth.modules.dataloader.metadata.MetaData] object containing the metadata for the dataset. If this is not provided it will be inferred from the dataset.
        missingness_strategy: The missingness strategy to use. Defaults to augmenting missing values in the data, see [the missingness strategies][nhssynth.modules.dataloader.missingness] for more information.
        impute_value: Only used when `missingness_strategy` is set to 'impute'. The value to use when imputing missing values in the data.

    After calling `MetaTransformer.apply()`, the following attributes and methods will be available:

    Attributes:
        typed_dataset (pd.DataFrame): The dataset with the dtypes applied.
        post_missingness_strategy_dataset (pd.DataFrame): The dataset with the missingness strategies applied.
        transformed_dataset (pd.DataFrame): The transformed dataset.
        single_column_indices (list[int]): The indices of the columns that were transformed into a single column.
        multi_column_indices (list[list[int]]): The indices of the columns that were transformed into multiple columns.

    **Methods:**

    - `get_typed_dataset()`: Returns the typed dataset.
    - `get_prepared_dataset()`: Returns the dataset with the missingness strategies applied.
    - `get_transformed_dataset()`: Returns the transformed dataset.
    - `get_multi_and_single_column_indices()`: Returns the indices of the columns that were transformed into one or multiple column(s).
    - `get_sdv_metadata()`: Returns the metadata in the correct format for SDMetrics.
    - `save_metadata()`: Saves the metadata to a file.
    - `save_constraint_graphs()`: Saves the constraint graphs to a file.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.apply_missingness_strategy` and `mt.transform` in sequence.
    This is the recommended way to use the MetaTransformer to ensure that it is fully instantiated for use downstream.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        metadata: Optional[MetaData] = None,
        missingness_strategy: Optional[str] = "augment",
        impute_value: Optional[Any] = None,
    ):
        self._raw_dataset: pd.DataFrame = dataset
        self._metadata: MetaData = metadata or MetaData(dataset)
        if missingness_strategy == "impute":
            assert (
                impute_value is not None
            ), "`impute_value` of the `MetaTransformer` must be specified (via the --impute flag) when using the imputation missingness strategy"
            self._impute_value = impute_value
        self._missingness_strategy = MISSINGNESS_STRATEGIES[missingness_strategy]

    @classmethod
    def from_path(cls, dataset: pd.DataFrame, metadata_path: str, **kwargs) -> Self:
        """
        Instantiates a MetaTransformer from a metadata file via a provided path.

        Args:
            dataset: The raw input DataFrame.
            metadata_path: The path to the metadata file.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData.from_path(dataset, metadata_path), **kwargs)

    @classmethod
    def from_dict(cls, dataset: pd.DataFrame, metadata: dict, **kwargs) -> Self:
        """
        Instantiates a MetaTransformer from a metadata dictionary.

        Args:
            dataset: The raw input DataFrame.
            metadata: A dictionary of raw metadata.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData(dataset, metadata), **kwargs)

    def drop_columns(self) -> None:
        """
        Drops columns from the dataset that are not in the `MetaData`.
        """
        self._raw_dataset = self._raw_dataset[self._metadata.columns]

    def _apply_rounding_scheme(self, working_column: pd.Series, rounding_scheme: float) -> pd.Series:
        """
        A rounding scheme takes the form of the smallest value that should be rounded to 0, i.e. 0.01 for 2dp.
        We first round to the nearest multiple in the standard way, through dividing, rounding and then multiplying.
        However, this can lead to floating point errors, so we then round to the number of decimal places required by the rounding scheme.

        e.g. `np.round(0.15 / 0.1) * 0.1` will erroneously return 0.1.

        Args:
            working_column: The column to apply the rounding scheme to.
            rounding_scheme: The rounding scheme to apply.

        Returns:
            The column with the rounding scheme applied.
        """
        working_column = np.round(working_column / rounding_scheme) * rounding_scheme
        return working_column.round(max(0, int(np.ceil(np.log10(1 / rounding_scheme)))))

    def _apply_dtype(
        self,
        working_column: pd.Series,
        column_metadata: MetaData.ColumnMetaData,
    ) -> pd.Series:
        """
        Given a `working_column`, the dtype specified in the `column_metadata` is applied to it.
         - Datetime columns are floored, and their format is inferred.
         - Rounding schemes are applied to numeric columns if specified.
         - Columns with missing values have their dtype converted to the pandas equivalent to allow for NA values.

        Args:
            working_column: The column to apply the dtype to.
            column_metadata: The metadata for the column.

        Returns:
            The column with the dtype applied.
        """
        dtype = column_metadata.dtype
        try:
            if dtype.kind == "M":
                working_column = pd.to_datetime(working_column, format=column_metadata.datetime_config.get("format"))
                if column_metadata.datetime_config.get("floor"):
                    working_column = working_column.dt.floor(column_metadata.datetime_config.get("floor"))
                    column_metadata.datetime_config["format"] = column_metadata._infer_datetime_format(working_column)
                return working_column
            else:
                if hasattr(column_metadata, "rounding_scheme") and column_metadata.rounding_scheme is not None:
                    working_column = self._apply_rounding_scheme(working_column, column_metadata.rounding_scheme)
                # If there are missing values in the column, we need to use the pandas equivalent of the dtype to allow for NA values
                if working_column.isnull().any() and dtype.kind in ["i", "u", "f"]:
                    return working_column.astype(dtype.name.capitalize())
                else:
                    return working_column.astype(dtype)
        except ValueError:
            raise ValueError(f"{sys.exc_info()[1]}\nError applying dtype '{dtype}' to column '{working_column.name}'")

    def apply_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `dataset`.

        Returns:
            The dataset with the dtypes applied.
        """
        working_data = data.copy()
        for column_metadata in self._metadata:
            working_data[column_metadata.name] = self._apply_dtype(working_data[column_metadata.name], column_metadata)
        return working_data

    def apply_missingness_strategy(self) -> pd.DataFrame:
        """
        Resolves missingness in the dataset via the `MetaTransformer`'s global missingness strategy or
        column-wise missingness strategies. In the case of the `AugmentMissingnessStrategy`, the missingness
        is not resolved, instead a new column / value is added for later transformation.

        Returns:
            The dataset with the missingness strategies applied.
        """
        working_data = self.typed_dataset.copy()
        for column_metadata in self._metadata:
            if not column_metadata.missingness_strategy:
                column_metadata.missingness_strategy = (
                    self._missingness_strategy(self._impute_value)
                    if hasattr(self, "_impute_value")
                    else self._missingness_strategy()
                )
            if not working_data[column_metadata.name].isnull().any():
                continue
            working_data = column_metadata.missingness_strategy.remove(working_data, column_metadata)
        return working_data

    # def apply_constraints(self) -> pd.DataFrame:
    #     working_data = self.post_missingness_strategy_dataset.copy()
    #     for constraint in self._metadata.constraints:
    #         working_data = constraint.apply(working_data)
    #     return working_data

    def _get_missingness_carrier(self, column_metadata: MetaData.ColumnMetaData) -> Union[pd.Series, Any]:
        """
        In the case of the `AugmentMissingnessStrategy`, a `missingness_carrier` has been determined for each column.
        For continuous columns this is an indicator column for the presence of NaN values.
        For categorical columns this is the value to be used to represent missingness as a category.

        Args:
            column_metadata: The metadata for the column.

        Returns:
            The missingness carrier for the column.
        """
        missingness_carrier = getattr(column_metadata.missingness_strategy, "missingness_carrier", None)
        if missingness_carrier in self.post_missingness_strategy_dataset.columns:
            return self.post_missingness_strategy_dataset[missingness_carrier]
        else:
            return missingness_carrier

    def transform(self) -> pd.DataFrame:
        """
        Prepares the dataset by applying each of the columns' transformers and recording the indices of the single and multi columns.

        Returns:
            The transformed dataset.
        """
        transformed_columns = []
        self.single_column_indices = []
        self.multi_column_indices = []
        col_counter = 0
        working_data = self.post_missingness_strategy_dataset.copy()

        # iteratively build the transformed df
        for column_metadata in tqdm(
            self._metadata, desc="Transforming data", unit="column", total=len(self._metadata.columns)
        ):
            missingness_carrier = self._get_missingness_carrier(column_metadata)
            transformed_data = column_metadata.transformer.apply(
                working_data[column_metadata.name], missingness_carrier
            )
            transformed_columns.append(transformed_data)

            # track single and multi column indices to supply to the model
            if isinstance(transformed_data, pd.DataFrame) and transformed_data.shape[1] > 1:
                num_to_add = transformed_data.shape[1]
                if not column_metadata.categorical:
                    self.single_column_indices.append(col_counter)
                    col_counter += 1
                    num_to_add -= 1
                self.multi_column_indices.append(list(range(col_counter, col_counter + num_to_add)))
                col_counter += num_to_add
            else:
                self.single_column_indices.append(col_counter)
                col_counter += 1

        return pd.concat(transformed_columns, axis=1)

    def apply(self) -> pd.DataFrame:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Returns:
            The transformed dataset.
        """
        self.drop_columns()
        self.typed_dataset = self.apply_dtypes(self._raw_dataset)
        self.post_missingness_strategy_dataset = self.apply_missingness_strategy()
        # self.constrained_dataset = self.apply_constraints()
        self.transformed_dataset = self.transform()
        return self.transformed_dataset

    def inverse_apply(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            dataset: The transformed dataset.

        Returns:
            The original dataset.
        """
        for column_metadata in self._metadata:
            dataset = column_metadata.transformer.revert(dataset)
        return self.apply_dtypes(dataset)

    def get_typed_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "typed_dataset"):
            raise ValueError(
                "The typed dataset has not yet been created. Call `mt.apply()` (or `mt.apply_dtypes()`) first."
            )
        return self.typed_dataset

    def get_prepared_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "prepared_dataset"):
            raise ValueError(
                "The prepared dataset has not yet been created. Call `mt.apply()` (or `mt.apply_missingness_strategy()`) first."
            )
        return self.prepared_dataset

    def get_transformed_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "transformed_dataset"):
            raise ValueError(
                "The prepared dataset has not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.transformed_dataset

    def get_multi_and_single_column_indices(self) -> tuple[list[int], list[int]]:
        """
        Returns the indices of the columns that were transformed into one or multiple column(s).

        Returns:
            A tuple containing the indices of the single and multi columns.
        """
        if not hasattr(self, "multi_column_indices") or not hasattr(self, "single_column_indices"):
            raise ValueError(
                "The single and multi column indices have not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.multi_column_indices, self.single_column_indices

    def get_sdv_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Calls the `MetaData` method to reformat its contents into the correct format for use with SDMetrics.

        Returns:
            The metadata in the correct format for SDMetrics.
        """
        return self._metadata.get_sdv_metadata()

    def save_metadata(self, path: pathlib.Path, collapse_yaml: bool = False) -> None:
        return self._metadata.save(path, collapse_yaml)

    def save_constraint_graphs(self, path: pathlib.Path) -> None:
        return self._metadata.constraints._output_graphs_html(path)
