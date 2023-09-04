import pathlib
import sys
from typing import Any, Callable, Optional, Self, Union

import pandas as pd
from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import *
from tqdm import tqdm


class MetaTransformer:
    """
    A metatransformer object that can wrap a [`BaseSingleTableSynthesizer`](https://docs.sdv.dev/sdv/single-table-dataset/modeling/synthesizers)
    from SDV. The metatransformer is responsible for transforming input dataset into a format that can be used by the model module, and transforming
    the module's output back to the original format of the input dataset.

    Args:
        metadata: A dictionary mapping column names to their metadata.

    Once instantiated via `mt = MetaTransformer(<parameters>)`, the following attributes will be available:

    Attributes:
        dtypes: A dictionary mapping each column to its specified pandas dtype (will infer from pandas defaults if this is missing).
        sdtypes: A dictionary mapping each column to the appropriate SDV-specific dataset type.
        transformers: A dictionary mapping each column to their assigned (if any) transformer.

    After preparing some dataset with the MetaTransformer, i.e. `transformed_dataset = mt.apply(dataset)`, the following attributes and methods will be available:

    Attributes:
        metatransformer (self.Synthesizer): An instanatiated `self.Synthesizer` object, ready to use on dataset.
        assembled_metadata (dict[str, dict[str, Any]]): A dictionary containing the formatted and complete metadata for the MetaTransformer.
        multi_column_indices (list[list[int]]): The groups of indices of one-hotted columns (i.e. each inner list contains all levels of one categorical).
        single_column_indices (list[int]): The indices of non-one-hotted columns.

    **Methods:**

    - `get_assembled_metadata()`: Returns the assembled metadata.
    - `get_sdtypes()`: Returns the sdtypes from the assembled metadata in the correct format for SDMetrics.
    - `get_multi_column_indices_and_single_column_indices()`: Returns the values of the MetaTransformer's `multi_column_indices` and `single_column_indices` attributes.
    - `inverse_apply(synthetic_data)`: Apply the inverse of the MetaTransformer to the given dataset.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.instaniate`, `mt.assemble`, `mt.prepare` and finally
    `mt.count_multi_column_indices_and_single_column_indices` in sequence on a given raw dataset. Along the way it assigns the attributes listed above. *This workflow is highly
    encouraged to ensure that the MetaTransformer is properly instantiated for use with the model module.*
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        metadata: Optional[MetaData] = None,
        missingness_strategy: Optional[str] = "augment",
        impute_value: Optional[Any] = None,
    ):
        self.raw_dataset: pd.DataFrame = dataset
        self.metadata: MetaData = metadata or MetaData(dataset)
        if missingness_strategy == "impute":
            assert (
                impute_value is not None
            ), "`impute_value` must be specified when using the imputation missingness strategy"
            self.missingness_strategy = self._impute_missingness_strategy_generator(impute_value)
        else:
            self.missingness_strategy = MISSINGNESS_STRATEGIES[missingness_strategy]

    def _impute_missingness_strategy_generator(self, impute_value: Any) -> Callable[[], ImputeMissingnessStrategy]:
        def _impute_missingness_strategy() -> ImputeMissingnessStrategy:
            return ImputeMissingnessStrategy(impute_value)

        return _impute_missingness_strategy

    @classmethod
    def from_path(cls, dataset: pd.DataFrame, metadata_path: str, **kwargs) -> Self:
        """
        Instantiates a MetaTransformer from a metadata file.

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
            metadata: The metadata dictionary.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData(dataset, metadata), **kwargs)

    def drop_columns(self) -> None:
        """
        Drops columns from the dataset that are not in the metadata.
        """
        self.raw_dataset = self.raw_dataset[self.metadata.columns]

    def _apply_rounding_scheme(self, working_column: pd.Series, rounding_scheme: float) -> pd.Series:
        working_column = np.round(working_column / rounding_scheme) * rounding_scheme
        return working_column.round(max(0, int(np.ceil(np.log10(1 / rounding_scheme)))))

    def _apply_dtype(
        self,
        working_column: pd.Series,
        column_metadata: MetaData.ColumnMetaData,
    ) -> pd.Series:
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
        except:
            raise ValueError(f"{sys.exc_info()[1]}\nError applying dtype '{dtype}' to column '{working_column.name}'")

    def apply_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `dataset`.

        Returns:
            The dataset with the dtypes applied.
        """
        working_data = data.copy()
        for column_metadata in self.metadata:
            working_data[column_metadata.name] = self._apply_dtype(working_data[column_metadata.name], column_metadata)
        return working_data

    def apply_missingness_strategy(self) -> pd.DataFrame:
        """
        Resolves missingness in the dataset via the `MetaTransformer`'s or column-wise missingness strategies.
        In the case of the `AugmentMissingnessStrategy`, the missingness is not resolved, instead a new
        column / value is added for later transformation.

        Returns:
            The dataset with the missingness strategies applied.
        """
        working_data = self.typed_dataset.copy()
        for column_metadata in self.metadata:
            if not column_metadata.missingness_strategy:
                column_metadata.missingness_strategy = self.missingness_strategy()
            if not working_data[column_metadata.name].isnull().any():
                continue
            working_data = column_metadata.missingness_strategy.remove(working_data, column_metadata)
        return working_data

    # def apply_constraints(self) -> pd.DataFrame:
    #     working_data = self.post_missingness_strategy_dataset.copy()
    #     for constraint in self.metadata.constraints:
    #         working_data = constraint.apply(working_data)
    #     return working_data

    def _get_missingness_carrier(self, column_metadata: MetaData.ColumnMetaData) -> Union[pd.Series, Any]:
        missingness_carrier = getattr(column_metadata.missingness_strategy, "missingness_carrier", None)
        if missingness_carrier in self.post_missingness_strategy_dataset.columns:
            return self.post_missingness_strategy_dataset[missingness_carrier]
        else:
            return missingness_carrier

    def transform(self) -> pd.DataFrame:
        """
        Prepares the dataset by processing it via the metatransformer.

        Returns:
            The transformed dataset.

        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        transformed_columns = []
        self.single_column_indices = []
        self.multi_column_indices = []
        col_counter = 0
        working_data = self.post_missingness_strategy_dataset.copy()

        print("")

        for column_metadata in tqdm(
            self.metadata, desc="Transforming data", unit="column", total=len(self.metadata.columns)
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
        self.typed_dataset = self.apply_dtypes(self.raw_dataset)
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
        for column_metadata in self.metadata:
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
        Returns the metadata in the correct format for SDMetrics.
        """
        return self.metadata.get_sdv_metadata()

    def save_metadata(self, path: pathlib.Path, collapse_yaml: bool = False) -> None:
        return self.metadata.save(path, collapse_yaml)

    def save_constraint_graphs(self, path: pathlib.Path) -> None:
        return self.metadata.constraints._output_graphs_html(path)
