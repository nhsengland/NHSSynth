import warnings
from typing import Any

import numpy as np
import pandas as pd
from nhssynth.common.constants import SDV_SYNTHESIZERS
from nhssynth.common.dicts import filter_dict
from nhssynth.modules.dataloader.metadata import get_sdtypes
from rdt.transformers import *
from sdv.metadata import SingleTableMetadata
from sdv.single_table.base import BaseSingleTableSynthesizer


# TODO should this be an @classmethod?
def get_transformer(d: dict) -> BaseTransformer | None:
    """
    Return a callable transformer object constructed from data in the given dictionary.

    Args:
        d: A dictionary containing the transformer data.

    Returns:
        An instantiated `BaseTransformer` if the dictionary contains valid transformer data, else None.
    """
    transformer_data = d.get("transformer", None)
    if isinstance(transformer_data, dict) and "name" in transformer_data:
        # Need to copy in case dicts are shared across columns, this can happen when reading a yaml with anchors
        transformer_data = transformer_data.copy()
        transformer_name = transformer_data.pop("name")
        return eval(transformer_name)(**transformer_data)
    elif isinstance(transformer_data, str):
        return eval(transformer_data)()
    else:
        return None


# TODO should this be an @classmethod?
def make_transformer_dict(transformer: BaseTransformer) -> dict[str, Any]:
    """
    Deconstruct a `transformer` into a dictionary of config.

    Args:
        transformer: A BaseTransformer object from RDT (SDV).

    Returns:
        A dictionary containing the transformer's name and arguments.
    """
    return {
        "name": type(transformer).__name__,
        **filter_dict(
            transformer.__dict__,
            {"output_properties", "random_states", "transform", "reverse_transform", "_dtype"},
        ),
    }


# TODO Can we come up with a way to instantiate this from the `model` module without needing to pickle and pass? Not high priority but would be nice to have
class MetaTransformer:
    """
    A metatransformer object that can wrap a [`BaseSingleTableSynthesizer`](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)
    from SDV. The metatransformer is responsible for transforming input data into a format that can be used by the model module, and transforming
    the module's output back to the original format of the input data.

    Args:
        metadata: A dictionary mapping column names to their metadata.
        allow_null_transformers: A flag indicating whether or not to allow null transformers on some / all columns.
        synthesizer: The `BaseSingleTableSynthesizer` class to use as the "host" for the MetaTransformer.

    Once instantiated via `mt = MetaTransformer(<parameters>)`, the following attributes will be available:

    Attributes:
        allow_null_transformers: A flag indicating whether or not to allow null transformers on some / all columns.
        Synthesizer: The `BaseSingleTableSynthesizer` host class.
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

    def __init__(self, metadata, allow_null_transformers, synthesizer) -> None:
        self.allow_null_transformers: bool = allow_null_transformers
        self.Synthesizer: BaseSingleTableSynthesizer = SDV_SYNTHESIZERS[synthesizer]
        self.dtypes: dict[str, dict[str, Any]] = {cn: cd.get("dtype", {}) for cn, cd in metadata.items()}
        self.sdtypes: dict[str, dict[str, Any]] = {
            cn: filter_dict(cd, {"dtype", "transformer"}) for cn, cd in metadata.items()
        }
        self.transformers: dict[str, BaseTransformer | None] = {cn: get_transformer(cd) for cn, cd in metadata.items()}

    def apply_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `data` and infers missing dtypes by reading pandas defaults.

        Args:
            data: The raw input DataFrame.

        Returns:
            The data with the dtypes applied.
        """
        if not all(self.dtypes.values()):
            warnings.warn(
                f"Incomplete metadata, detecting missing `dtype`s for column(s): {[k for k, v in self.dtypes.items() if not v]} automatically...",
                UserWarning,
            )
            self.dtypes.update({cn: data[cn].dtype for cn, cv in self.dtypes.items() if not cv})
        return data.astype(self.dtypes)

    def _instantiate_ohe_component_transformers(
        self, transformers: dict[str, BaseTransformer | None]
    ) -> dict[str, BaseTransformer]:
        """
        Instantiates a OneHotEncoder for each resulting `*.component` column that arises from a ClusterBasedNormalizer.

        Args:
            transformers: A dictionary mapping column names to their assigned transformers.

        Returns:
            A dictionary mapping each `*.component` column to a OneHotEncoder.
        """
        return {
            f"{cn}.component": OneHotEncoder()
            for cn, transformer in transformers.items()
            if transformer.get_name() == "ClusterBasedNormalizer"
        }

    def instantiate(self, data: pd.DataFrame) -> BaseSingleTableSynthesizer:
        """
        Instantiates a `self.Synthesizer` object from the given metadata and data. Infers missing metadata (sdtypes and transformers).

        Args:
            data: The input DataFrame.

        Returns:
            A fully instantiated `self.Synthesizer` object and a transformer for the `*.component` columns.

        Raises:
            UserWarning: If the metadata is incomplete (and `self.allow_null_transformers` is `False`) in the case of missing transformer metadata.
        """
        if all(self.sdtypes.values()):
            metadata = SingleTableMetadata.load_from_dict({"columns": self.sdtypes})
        else:
            warnings.warn(
                f"Incomplete metadata, detecting missing `sdtype`s for column(s): {[k for k, v in self.sdtypes.items() if not v]} automatically...",
                UserWarning,
            )
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
            for column_name, values in self.sdtypes.items():
                if values:
                    metadata.update_column(column_name=column_name, **values)
        if not all(self.transformers.values()) and not self.allow_null_transformers:
            warnings.warn(
                f"Incomplete metadata, detecting missing `transformers`s for column(s): {[k for k, v in self.transformers.items() if not v]} automatically...",
                UserWarning,
            )
        synthesizer = self.Synthesizer(metadata)
        synthesizer.auto_assign_transformers(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synthesizer.update_transformers(
                self.transformers if self.allow_null_transformers else {k: v for k, v in self.transformers.items() if v}
            )
        # TODO this is a hacky way to get the component columns we want to apply OneHotEncoder to
        component_transformer = self._instantiate_ohe_component_transformers(synthesizer.get_transformers())
        return synthesizer, component_transformer

    def _get_dtype(self, cn: str) -> str | np.dtype:
        """Returns the dtype for the given column name `cn`."""
        return self.dtypes[cn].name if not isinstance(self.dtypes[cn], str) else self.dtypes[cn]

    def assemble(self) -> dict[str, dict[str, Any]]:
        """
        Rearranges the dtype, sdtype and transformer metadata into a consistent format ready for output.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas data type for the column
                - sdtype: The SDV-specific data type for the column.
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
                "dtype": self._get_dtype(cn),
            }
            for cn, cd in self.metatransformer.metadata.columns.items()
        }

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data by processing it via the metatransformer.

        Args:
            data: The data to fit and apply the transformer to.

        Returns:
            The transformed data.

        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        if not self.metatransformer:
            raise ValueError(
                "The metatransformer has not yet been instantiated. Call `mt.apply(data)` first (or `mt.instantiate(data)`)."
            )
        prepared_data = self.metatransformer.preprocess(data)
        # TODO this is kind of a hacky way to solve the component column problem
        for cn, transformer in self.component_transformer.items():
            prepared_data = transformer.fit_transform(prepared_data, cn)
        return prepared_data

    def count_onehots_and_singles(self, data: pd.DataFrame) -> tuple[list[list[int]], list[int]]:
        """
        Uses the assembled metadata to identify and record the indices of one-hotted column groups.
        Also records the indices of non-one-hotted columns in a separate list.

        Args:
            data: The data to extract column indices from.

        Returns:
            A pair of lists:
                - One-hotted column index groups (i.e. one inner list with all corresponding indices per categorical variable)
                - Non-one-hotted column indices
        """
        if not self.assembled_metadata:
            self.assembled_metadata = self.assemble()
        onehot_idxs = []
        single_idxs = []
        for cn, cd in self.assembled_metadata.items():
            if cd["transformer"].get("name") == "OneHotEncoder":
                onehot_idxs.append(data.columns.get_indexer(data.filter(like=cn + ".value").columns).tolist())
            elif cd["transformer"].get("name") == "ClusterBasedNormalizer":
                onehot_idxs.append(data.columns.get_indexer(data.filter(like=cn + ".component.value").columns).tolist())
                single_idxs.append(data.columns.get_loc(cn + ".normalized"))
            elif cd["transformer"].get("name") != "RegexGenerator":
                single_idxs.append(data.columns.get_loc(cn))
        if not onehot_idxs:
            onehot_idxs.append([])
        return onehot_idxs, single_idxs

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Args:
            data: The DataFrame to transform.

        Returns:
            The transformed data.
        """
        typed_data = self.apply_dtypes(data)
        self.metatransformer, self.component_transformer = self.instantiate(typed_data)
        self.assembled_metadata = self.assemble()
        prepared_data = self.prepare(typed_data)
        self.onehots, self.singles = self.count_onehots_and_singles(prepared_data)
        return typed_data, prepared_data

    def get_assembled_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Returns the assembled metadata for the transformer.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas data type for the column
                - sdtype: The SDV-specific data type for the column.
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.

        Raises:
            ValueError: If the metadata has not yet been assembled.
        """
        if not hasattr(self, "assembled_metadata"):
            raise ValueError("Metadata has not yet been assembled. Call `my.apply(data)` (or `mt.assemble()`) first.")
        return self.assembled_metadata

    def get_sdtypes(self) -> dict[str, dict[str, dict[str, str]]]:
        """
        Returns the sdtypes extracted from the assembled metadata for SDMetrics.

        Returns:
            A dictionary mapping column names to sdtypes.

        Raises:
            ValueError: If the metadata has not yet been assembled.
        """
        if not hasattr(self, "assembled_metadata"):
            raise ValueError("Metadata has not yet been assembled. Call `my.apply(data)` (or `mt.assemble()`) first.")
        return get_sdtypes(self.assembled_metadata)

    def get_onehots_and_singles(self) -> tuple[list[list[int]], list[int]]:
        """
        Get the values of the MetaTransformer's `onehots` and `singles` attributes.

        Returns:
            A pair of lists:
                - One-hotted column index groups (i.e. one inner list with all corresponding indices per categorical variable)
                - Non-one-hotted column indices

        Raises:
            ValueError: If `self.onehots` and `self.singles` have yet to be counted.
        """
        if not hasattr(self, "onehots") or not hasattr(self, "singles"):
            raise ValueError(
                "Some metadata is missing. Call `mt.apply(data)` first (or `mt.count_onehots_and_singles(data)`)."
            )
        return self.onehots, self.singles

    def inverse_apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            data: The transformed data.

        Returns:
            The original data.

        Raises:
            ValueError: If the metatransformer has not yet been instantiated.
        """
        if not hasattr(self, "metatransformer"):
            raise ValueError(
                "The metatransformer has not yet been instantiated. Call `mt.apply(data)` first (or `mt.instantiate(data)`)."
            )
        for transformer in self.component_transformer.values():
            data = transformer.reverse_transform(data)
        return self.metatransformer._data_processor.reverse_transform(data)
