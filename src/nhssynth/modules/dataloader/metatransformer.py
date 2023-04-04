import warnings
from typing import Any

import numpy as np
import pandas as pd
from nhssynth.common.constants import SDV_SYNTHESIZER_CHOICES
from nhssynth.common.dicts import filter_dict
from rdt import HyperTransformer
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
    A metatransformer object that can wrap either a `HyperTransformer` from RDT or a `BaseSingleTableSynthesizer` from SDV. The metatransformer
    is responsible for transforming input data into a format that can be used by the model module, and transforming the module's output back to
    the original format of the input data.

    Args:
        metadata: A dictionary mapping column names to their metadata.
        sdv_workflow: A flag indicating whether or not to use the SDV workflow.
        allow_null_transformers: A flag indicating whether or not to allow null transformers on some / all columns.
        synthesizer: The `BaseSingleTableSynthesizer` class to use within the SDV workflow.

    Once instantiated via `mt = MetaTransformer(<parameters>)`, the following attributes will be available:

    Attributes:
        sdv_workflow: A flag indicating whether or not to use the SDV workflow.
        allow_null_transformers: A flag indicating whether or not to allow null transformers on some / all columns.
        Synthesizer: The `BaseSingleTableSynthesizer` class to use within the SDV workflow.
        dtypes: A dictionary mapping each column to its specified pandas dtype (will infer from pandas defaults if this is missing).
        sdtypes: A dictionary mapping each column to the appropriate SDV-specific data type.
        transformers: A dictionary mapping each column to their assigned (if any) transformer.

    After preparing some data with the MetaTransformer, i.e. `prepared_data = mt.apply(data)`, the following attributes and methods will be available:

    Attributes:
        metatransformer (HyperTransformer | self.Synthesizer): An instanatiated `HyperTransformer` or `self.Synthesizer` object, ready to use on data.
        assembled_metadata (dict[str, dict[str, Any]]): A dictionary containing the formatted and complete metadata for the MetaTransformer.
        categoricals (dict[str, int]): A dictionary mapping categorical columns to the number of levels in the variable.
        num_continuous (int): The number of continuous columns.

    **Methods:**

    - `get_assembled_metadata()`: Return the assembled metadata.
    - `order(prepared_data)`: Order the prepared data according to the MetaTransformer's `categoricals` and `num_continuous` attributes.
    - `inverse_apply(synthetic_data)`: Apply the inverse of the MetaTransformer to the given data.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.instaniate`, `mt.assemble`, `mt.infer_categoricals`,
    `mt.infer_num_continuous`, and finally `mt.prepare` in sequence on a given raw dataset. Along the way it assigns the attributes listed above.
    This workflow is highly encouraged to ensure that the MetaTransformer is properly instantiated for use with the model module.
    """

    def __init__(self, metadata, sdv_workflow, allow_null_transformers, synthesizer) -> None:
        # self.metadata: dict[str, dict[str, Any]] = metadata
        self.sdv_workflow: bool = sdv_workflow
        self.allow_null_transformers: bool = allow_null_transformers
        self.Synthesizer: BaseSingleTableSynthesizer = SDV_SYNTHESIZER_CHOICES[synthesizer]
        # TODO think about whether these belong here
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

    def _instantiate_synthesizer(self, data: pd.DataFrame) -> BaseSingleTableSynthesizer:
        """
        Instantiates a `self.Synthesizer` object from the given metadata and data. Infers missing metadata (sdtypes and transformers).

        Args:
            data: The input DataFrame.

        Returns:
            A fully instantiated `self.Synthesizer` object.

        Raises:
            UserWarning: If the metadata is incomplete and `self.allow_null_transformers` is `False`.
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
        synthesizer.update_transformers(
            self.transformers if self.allow_null_transformers else {k: v for k, v in self.transformers.items() if v}
        )
        return synthesizer

    def _instantiate_hypertransformer(self, data: pd.DataFrame) -> HyperTransformer:
        """
        Instantiates a `HyperTransformer` object from the metadata and given data. Infers missing metadata (sdtypes and transformers).

        Args:
            data: The input DataFrame.

        Returns:
            A fully instantiated `HyperTransformer` object.

        Raises:
            UserWarning: If the metadata is incomplete.
        """
        ht = HyperTransformer()
        if all(self.sdtypes.values()) and (all(self.transformers.values()) or self.allow_null_transformers):
            ht.set_config(
                config={
                    "sdtypes": {k: v["sdtype"] for k, v in self.sdtypes.items()},
                    "transformers": self.transformers,
                }
            )
        else:
            warnings.warn(
                f"Incomplete metadata, detecting missing{(' `sdtype`s for column(s): ' + str([k for k, v in self.sdtypes.items() if not v])) if not all(self.sdtypes.values()) else ''}{(' `transformer`s for column(s): ' + str([k for k, v in self.transformers.items() if not v])) if not all(self.transformers.values()) and not self.allow_null_transformers else ''} automatically...",
                UserWarning,
            )
            ht.detect_initial_config(data)
            ht.update_sdtypes({k: v["sdtype"] for k, v in self.sdtypes.items() if v})
            ht.update_transformers(
                self.transformers if self.allow_null_transformers else {k: v for k, v in self.transformers.items() if v}
            )
        return ht

    def instantiate(self, data: pd.DataFrame) -> BaseSingleTableSynthesizer | HyperTransformer:
        """
        Calls the appropriate instantiation method based on the value of `self.sdv_workflow`.

        Args:
            data: The input DataFrame.

        Returns:
            A fully instantiated `self.Synthesizer` or `HyperTransformer` object.
        """
        if self.sdv_workflow:
            return self._instantiate_synthesizer(data)
        else:
            return self._instantiate_hypertransformer(data)

    def _get_dtype(self, cn: str) -> str | np.dtype:
        """Returns the dtype for the given column name `cn`."""
        return self.dtypes[cn].name if not isinstance(self.dtypes[cn], str) else self.dtypes[cn]

    def assemble(self) -> dict[str, dict[str, Any]]:
        """
        Rearranges the dtype, sdtype and transformer metadata into a consistent format regardless of the value of `self,sdv_workflow`

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - dtype: The pandas data type for the column
                - sdtype: The SDV-specific data type for the column.
                - transformer: A dictionary containing information about the transformer used for the column (if any). The dictionary has the following keys:
                    - name: The name of the transformer.
                    - Any other properties of the transformer that we want to record in output.
        """
        if self.sdv_workflow:
            sdmetadata = self.metatransformer.metadata
            transformers = self.metatransformer.get_transformers()
            return {
                cn: {
                    **cd,
                    "transformer": make_transformer_dict(transformers[cn]) if transformers[cn] else None,
                    "dtype": self._get_dtype(cn),
                }
                for cn, cd in sdmetadata.columns.items()
            }
        else:
            config = self.metatransformer.get_config()
            return {
                cn: {
                    "sdtype": cd,
                    "transformer": make_transformer_dict(config["transformers"][cn])
                    if config["transformers"][cn]
                    else None,
                    "dtype": self._get_dtype(cn),
                }
                for cn, cd in config["sdtypes"].items()
            }

    def infer_categoricals(self, data: pd.DataFrame) -> dict[str, int]:
        """
        Uses the (assembled) metadata to identify the categorical columns from the data and count the number of levels each categorical variable has.

        Args:
            data: The data to infer the categorical columns from using `self.assembled_metadata`

        Returns:
            A dictionary mapping column names to the number of unique values in the column (if the column is categorical).
        """
        return {cn: data[cn].nunique() for cn, cd in self.assembled_metadata.items() if cd["sdtype"] == "categorical"}

    def infer_num_continuous(self) -> int:
        """
        Uses the (assembled) metadata to infers the number of continuous columns in the data.

        Returns:
            The number of continuous columns
        """
        return sum([cd["sdtype"] != "categorical" for cd in self.assembled_metadata.values()])

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data by processing it via the metatransformer.

        Args:
            data: The data to fit and apply the transformer to.
        """
        if self.sdv_workflow:
            return self.metatransformer.preprocess(data)
        else:
            return self.metatransformer.fit_transform(data)

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Args:
            data: The DataFrame to transform.

        Returns:
            The transformed data.
        """
        typed_data = self.apply_dtypes(data)
        self.metatransformer = self.instantiate(typed_data)
        self.assembled_metadata = self.assemble()
        self.categoricals = self.infer_categoricals(typed_data)
        self.num_continuous = self.infer_num_continuous()
        return self.prepare(typed_data)

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
        if not self.assembled_metadata:
            raise ValueError(
                "Metadata has not yet been assembled. Call `MetaTransformer.apply` (or `MetaTransformer.assemble`) first."
            )
        return self.assembled_metadata

    def order(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[int], int]:
        """
        Orders the data based on the inferred categorical and continuous columns.

        Args:
            data: The data to order.

        Returns:
            A tuple containing the ordered data, a list of the number of unique values for each categorical column, and the number of continuous columns.

        Raises:
            ValueError: If the metadata has not yet been finalised and assembled or `categoricals` and/or `num_continuous` have yet to be inferred.
            ValueError: If a column in `self.assembled_metadata` is not found in the passed data.
        """
        if not self.categoricals or not self.num_continuous or not self.assembled_metadata:
            raise ValueError(
                "Some metadata is missing. Call `mt.apply(data)` first (or `mt.[assemble(),infer_categoricals(data),infer_num_continuous()]` in sequence)."
            )
        categorical_ordering = []
        continuous_ordering = []
        for cn, cd in self.assembled_metadata.items():
            if cd["transformer"] and cd["transformer"]["name"] == "OneHotEncoder":
                idx = data.columns.get_loc(cn + ".value0")
                categorical_ordering += [*range(idx, idx + self.categoricals[cn])]
            elif cn not in data.columns:
                raise ValueError(f"The {cn} column was not found in the passed data.")
            else:
                continuous_ordering.append(data.columns.get_loc(cn))
        ordering = categorical_ordering + continuous_ordering
        return data.iloc[:, ordering], list(self.categoricals.values()), self.num_continuous

    def inverse_apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            data: The transformed data.

        Returns:
            The original data.
        """
        if self.sdv_workflow:
            return self.metatransformer.reverse_transform(data)
        else:
            return self.metatransformer.reverse_transform(data)
