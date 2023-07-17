import sys
from typing import Any, Optional

import pandas as pd
from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import *
from nhssynth.modules.dataloader.transformers.utils import make_transformer_dict


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
        onehots (list[list[int]]): The groups of indices of one-hotted columns (i.e. each inner list contains all levels of one categorical).
        singles (list[int]): The indices of non-one-hotted columns.

    **Methods:**

    - `get_assembled_metadata()`: Returns the assembled metadata.
    - `get_sdtypes()`: Returns the sdtypes from the assembled metadata in the correct format for SDMetrics.
    - `get_onehots_and_singles()`: Returns the values of the MetaTransformer's `onehots` and `singles` attributes.
    - `inverse_apply(synthetic_data)`: Apply the inverse of the MetaTransformer to the given dataset.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.instaniate`, `mt.assemble`, `mt.prepare` and finally
    `mt.count_onehots_and_singles` in sequence on a given raw dataset. Along the way it assigns the attributes listed above. *This workflow is highly
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
        assert missingness_strategy in MISSINGNESS_STRATEGIES, (
            f"Invalid missingness strategy '{missingness_strategy}'. "
            f"Must be one of {list(MISSINGNESS_STRATEGIES.keys())}"
        )
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
                    if hasattr(self, "impute_value")
                    else self.missingness_strategy()
                )
            if not working_data[column_metadata.name].isnull().any():
                continue
            working_data = column_metadata.missingness_strategy.remove(working_data, column_metadata)
        return working_data

    def assemble(self) -> dict[str, dict[str, Any]]:
        """
        Rearranges the dtype, sdtype and transformer metadata into a consistent format ready for output.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas dataset type for the column
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.
        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        if not self.metatransformer:
            raise ValueError(
                "The metatransformer has not yet been instantiated. Call `mt.apply(dataset)` first (or `mt.instantiate(dataset)`)."
            )
        transformers = self.metatransformer.get_transformers()
        return {
            cn: {
                **cd,
                "transformer": make_transformer_dict(transformers[cn]) if transformers[cn] else None,
                "dtype": self.metadata[cn].dtype,
            }
            for cn, cd in self.metatransformer.metadata.columns.items()
        }

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
        self.single_idxs = []
        self.multi_idxs = []
        col_counter = 0
        working_data = self.prepared_dataset.copy()
        for column_metadata in self.metadata:
            # TODO is there a nicer way of doing this, the transformer and augment strategy create a chicken and egg problem
            if hasattr(column_metadata.missingness_strategy, "missing_column") and not column_metadata.categorical:
                transformed_data = column_metadata.transformer.apply(
                    working_data[column_metadata.name],
                    working_data[column_metadata.missingness_strategy.missing_column],
                )
            else:
                transformed_data = column_metadata.transformer.apply(working_data[column_metadata.name])
            transformed_columns.append(transformed_data)
            if isinstance(transformed_data, pd.DataFrame) and transformed_data.shape[1] > 1:
                self.multi_idxs.append(list(range(col_counter, col_counter + transformed_data.shape[1])))
                col_counter += transformed_data.shape[1]
            else:
                self.single_idxs.append(col_counter)
                col_counter += 1
        return pd.concat(transformed_columns, axis=1)

    def apply(self) -> None:
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
        print(self.transformed_dataset.columns)

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
        return self.get_prepared_dataset

    def get_transformed_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "transformed_dataset"):
            raise ValueError(
                "The prepared dataset has not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.transformed_dataset

    def get_assembled_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Returns the assembled metadata for the transformer.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas dataset type for the column
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.

        Raises:
            ValueError: If the metadata has not yet been assembled.
        """
        if not hasattr(self, "assembled_metadata"):
            raise ValueError(
                "MetaData has not yet been assembled. Call `mt.apply(dataset)` (or `mt.assemble()`) first."
            )
        return self.assembled_metadata

    def inverse_apply(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            dataset: The transformed dataset.

        Returns:
            The original dataset.
        """
        for transformer in self.component_transformer.values():
            dataset = transformer.reverse_transform(dataset)
        return self.metatransformer._data_processor.reverse_transform(dataset)
