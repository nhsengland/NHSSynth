from typing import Any, Optional

import pandas as pd
from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import *
from nhssynth.modules.dataloader.transformers.utils import make_transformer_dict


# TODO Can we come up with a way to instantiate this from the `model` module without needing to pickle and pass? Not high priority but would be nice to have
class MetaTransformer:
    """
    A metatransformer object that can wrap a [`BaseSingleTableSynthesizer`](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)
    from SDV. The metatransformer is responsible for transforming input data into a format that can be used by the model module, and transforming
    the module's output back to the original format of the input data.

    Args:
        metadata: A dictionary mapping column names to their metadata.

    Once instantiated via `mt = MetaTransformer(<parameters>)`, the following attributes will be available:

    Attributes:
        dtypes: A dictionary mapping each column to its specified pandas dtype (will infer from pandas defaults if this is missing).
        sdtypes: A dictionary mapping each column to the appropriate SDV-specific data type.
        transformers: A dictionary mapping each column to their assigned (if any) transformer.

    After preparing some data with the MetaTransformer, i.e. `prepared_data = mt.apply(data)`, the following attributes and methods will be available:

    Attributes:
        metatransformer (self.Synthesizer): An instanatiated `self.Synthesizer` object, ready to use on data.
        assembled_metadata (dict[str, dict[str, Any]]): A dictionary containing the formatted and complete metadata for the MetaTransformer.
        onehots (list[list[int]]): The groups of indices of one-hotted columns (i.e. each inner list contains all levels of one categorical).
        singles (list[int]): The indices of non-one-hotted columns.

    **Methods:**

    - `get_assembled_metadata()`: Returns the assembled metadata.
    - `get_sdtypes()`: Returns the sdtypes from the assembled metadata in the correct format for SDMetrics.
    - `get_onehots_and_singles()`: Returns the values of the MetaTransformer's `onehots` and `singles` attributes.
    - `inverse_apply(synthetic_data)`: Apply the inverse of the MetaTransformer to the given data.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.instaniate`, `mt.assemble`, `mt.prepare` and finally
    `mt.count_onehots_and_singles` in sequence on a given raw dataset. Along the way it assigns the attributes listed above. *This workflow is highly
    encouraged to ensure that the MetaTransformer is properly instantiated for use with the model module.*
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[MetaData] = {},
        missingness_strategy: Optional[GenericMissingnessStrategy] = None,
    ):
        self.data: pd.DataFrame = data
        if metadata:
            assert metadata.columns == data.columns, "`metadata` and `data` must refer to the same columns"
            self.metadata: MetaData = metadata
        else:
            self.metadata: MetaData = MetaData(data)
        self.missingness_strategy = missingness_strategy

    @classmethod
    def from_path(cls, data: pd.DataFrame, metadata_path: str, **kwargs):
        """
        Instantiates a MetaTransformer from a metadata file.

        Args:
            data: The raw input DataFrame.
            metadata_path: The path to the metadata file.

        Returns:
            A MetaTransformer object.
        """
        return cls(data, MetaData.load(metadata_path, data), **kwargs)

    @classmethod
    def from_dict(cls, data: pd.DataFrame, metadata: dict, **kwargs):
        """
        Instantiates a MetaTransformer from a metadata dictionary.

        Args:
            data: The raw input DataFrame.
            metadata_dict: The metadata dictionary.

        Returns:
            A MetaTransformer object.
        """
        return cls(data, MetaData(data, metadata), **kwargs)

    def apply_dtypes(self) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `data` and infers missing dtypes by reading pandas defaults.

        Args:
            data: The raw input DataFrame.

        Returns:
            The data with the dtypes applied.
        """
        typed_data = self.data.copy()
        for cn in self.columns:
            dtype = self.metadata[cn].dtype
            if dtype.kind == "M":
                typed_data[cn] = pd.to_datetime(typed_data[cn], **self.metadata[cn].datetime_config)
            elif typed_data[cn].isnull().any():
                typed_data[cn] = typed_data[cn].astype(dtype.name.capitalize())
            else:
                typed_data[cn] = typed_data[cn].astype(dtype)
        return typed_data

    def resolve_missingness_strategy(self) -> pd.DataFrame:
        if not self.missingness_strategy:
            return self.data
        if isinstance(self.missingness_strategy, DropMissingnessStrategy):
            self.data = self.data.dropna()
        else:
            for cn in self.metadata.columns:
                self.metadata[cn].transformer.missingness_strategy = self.missingness_strategy
        return self.data

    def assemble(self) -> dict[str, dict[str, Any]]:
        """
        Rearranges the dtype, sdtype and transformer metadata into a consistent format ready for output.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas data type for the column
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.
        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        if not self.metatransformer:
            raise ValueError(
                "The metatransformer has not yet been instantiated. Call `mt.apply(data)` first (or `mt.instantiate(data)`)."
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
        Prepares the data by processing it via the metatransformer.

        Args:
            data: The data to fit and apply the transformer to.

        Returns:
            The transformed data.

        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        transformed_columns = []
        self.single_idxs = []
        self.multi_idxs = []
        col_counter = 0
        to_transform = self.typed_data.copy()
        for cn in self.columns:
            transformed_data = self.metadata[cn].transformer.transform(to_transform[cn])
            transformed_columns.append(transformed_data)
            if isinstance(transformed_data, pd.DataFrame) and transformed_data.shape[1] > 1:
                self.categorical_idxs.append(list(range(col_counter, col_counter + transformed_data.shape[1])))
                col_counter += transformed_data.shape[1]
            else:
                self.single_idxs.append(col_counter)
                col_counter += 1
        return pd.concat([transformed_columns], axis=1)

    def apply(self) -> None:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Args:
            data: The DataFrame to transform.

        Returns:
            The transformed data.
        """
        self.typed_data = self.apply_dtypes()
        self.typed_data = self.resolve_missingness_strategy(self.typed_data)
        self.prepared_data = self.transform()

    def get_typed_data(self) -> pd.DataFrame:
        """
        Returns the typed data.

        Returns:
            The typed data.
        """
        if not hasattr(self, "typed_data"):
            raise ValueError(
                "The typed data has not yet been created. Call `mt.apply()` (or `mt.apply_dtypes()`) first."
            )
        return self.typed_data

    def get_prepared_data(self) -> pd.DataFrame:
        """
        Returns the prepared data.

        Returns:
            The prepared data.
        """
        if not hasattr(self, "prepared_data"):
            raise ValueError(
                "The prepared data has not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.prepared_data

    def get_assembled_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Returns the assembled metadata for the transformer.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas data type for the column
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.

        Raises:
            ValueError: If the metadata has not yet been assembled.
        """
        if not hasattr(self, "assembled_metadata"):
            raise ValueError("MetaData has not yet been assembled. Call `mt.apply(data)` (or `mt.assemble()`) first.")
        return self.assembled_metadata

    def inverse_apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            data: The transformed data.

        Returns:
            The original data.
        """
        for transformer in self.component_transformer.values():
            data = transformer.reverse_transform(data)
        return self.metatransformer._data_processor.reverse_transform(data)
