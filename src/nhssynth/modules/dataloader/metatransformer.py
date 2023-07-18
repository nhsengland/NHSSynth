import pathlib
import sys
from typing import Any, Optional

import pandas as pd
from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import *
from tqdm import tqdm


# TODO Can we come up with a way to instantiate this from the `model` module without needing to pickle and pass? Not high priority but would be nice to have
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
        if metadata:
            assert metadata.columns.equals(dataset.columns), "`dataset`'s columns must match those in `metadata`"
            self.metadata: MetaData = metadata
        else:
            self.metadata: MetaData = MetaData(dataset)
        if missingness_strategy == "impute":
            assert (
                impute_value is not None
            ), "`impute_value` must be specified when using the imputation missingness strategy"
            self.impute_value: Any = impute_value
        self.missingness_strategy: GenericMissingnessStrategy = MISSINGNESS_STRATEGIES[missingness_strategy]

    @classmethod
    def from_path(cls, dataset: pd.DataFrame, metadata_path: str, **kwargs):
        """
        Instantiates a MetaTransformer from a metadata file.

        Args:
            dataset: The raw input DataFrame.
            metadata_path: The path to the metadata file.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData.load(metadata_path, dataset), **kwargs)

    @classmethod
    def from_dict(cls, dataset: pd.DataFrame, metadata: dict, **kwargs):
        """
        Instantiates a MetaTransformer from a metadata dictionary.

        Args:
            dataset: The raw input DataFrame.
            metadata_dict: The metadata dictionary.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData(dataset, metadata), **kwargs)

    def apply_dtypes(self) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `dataset` and infers missing dtypes by reading pandas defaults.

        Args:
            dataset: The raw input DataFrame.

        Returns:
            The dataset with the dtypes applied.
        """
        working_data = self.raw_dataset.copy()
        for column_metadata in self.metadata:
            dtype = column_metadata.dtype
            cn = column_metadata.name
            try:
                if dtype.kind == "M":
                    working_data[cn] = pd.to_datetime(working_data[cn], **column_metadata.datetime_config)
                elif working_data[cn].isnull().any() and dtype.kind in ["i", "u", "f"]:
                    working_data[cn] = working_data[cn].astype(dtype.name.capitalize())
                else:
                    working_data[cn] = working_data[cn].astype(dtype)
            except:
                # Print the error message along with the column name to make it easier to debug
                raise ValueError(f"{sys.exc_info()[1]}\nError applying dtype '{dtype}' to column '{cn}'")
        return working_data

    def apply_missingness_strategy(self) -> pd.DataFrame:
        """
        Applies the missingness strategy to the dataset.

        Args:
            dataset: The dataset to apply the missingness strategy to.

        Returns:
            The dataset with the missingness strategy applied.
        """
        working_data = self.typed_dataset.copy()
        for column_metadata in self.metadata:
            if not column_metadata.missingness_strategy:
                column_metadata.missingness_strategy = (
                    self.missingness_strategy(self.impute_value)
                    if isinstance(self.missingness_strategy, ImputeMissingnessStrategy)
                    else self.missingness_strategy()
                )
            if not working_data[column_metadata.name].isnull().any():
                continue
            working_data = column_metadata.missingness_strategy.remove(working_data, column_metadata)
            if column_metadata.dtype.kind in ["i", "u", "f"] and not isinstance(
                column_metadata.missingness_strategy, AugmentMissingnessStrategy
            ):
                working_data[column_metadata.name] = working_data[column_metadata.name].astype(
                    column_metadata.dtype.name.lower()
                )
        return working_data

    def transform(self) -> pd.DataFrame:
        """
        Prepares the dataset by processing it via the metatransformer.

        Args:
            dataset: The dataset to fit and apply the transformer to.

        Returns:
            The transformed dataset.

        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        transformed_columns = []
        self.single_column_indices = []
        self.multi_column_indices = []
        col_counter = 0
        working_data = self.prepared_dataset.copy()

        print("")

        for column_metadata in tqdm(
            self.metadata, desc="Transforming data", unit="column", total=len(self.metadata.columns)
        ):
            # TODO is there a nicer way of doing this, the transformer and augment strategy create a chicken and egg problem
            missingness_carrier = getattr(column_metadata.missingness_strategy, "missingness_carrier", None)
            if missingness_carrier in working_data.columns:
                missingness_carrier = working_data[missingness_carrier]
            transformed_data = column_metadata.transformer.apply(
                working_data[column_metadata.name], missingness_carrier
            )
            if column_metadata.dtype.kind in ["f", "i", "u"]:
                if isinstance(transformed_data, pd.DataFrame):
                    transformed_data = transformed_data.apply(lambda x: x.astype(column_metadata.dtype.name.lower()))
                else:
                    transformed_data = transformed_data.astype(column_metadata.dtype.name.lower())
            transformed_columns.append(transformed_data)
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

        print("")

        return pd.concat(transformed_columns, axis=1)

    def apply(self) -> pd.DataFrame:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Args:
            dataset: The DataFrame to transform.

        Returns:
            The transformed dataset.
        """
        self.typed_dataset = self.apply_dtypes()
        self.prepared_dataset = self.apply_missingness_strategy()
        self.transformed_dataset = self.transform()
        return self.transformed_dataset

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

    def save_metadata(self, path: pathlib.Path, collapse_yaml: bool = False) -> None:
        return self.metadata.save(path, collapse_yaml)

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
        return dataset
