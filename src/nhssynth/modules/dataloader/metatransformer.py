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
    Return a callable transformer object extracted from the given dictionary.

    Args:
        d: A dictionary containing the transformer data.

    Returns:
        A callable object (transformer) if the dictionary contains transformer data, else None.
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
def make_transformer_dict(transformer: BaseTransformer | None) -> dict:
    """
    Deconstruct an instance of a transformer (if one is present) into a dictionary of config.

    Args:
        transformer: A BaseTransformer object from RDT (SDV).

    Returns:
        A dictionary containing the transformer name and arguments.
    """
    if transformer:
        return {
            "name": type(transformer).__name__,
            **filter_dict(
                transformer.__dict__,
                {"output_properties", "random_states", "transform", "reverse_transform", "_dtype"},
            ),
        }
    else:
        return None


# TODO Can we come up with a way to instantiate this from the `model` module without needing to pickle and pass? Not high priority but would be nice to have
class MetaTransformer:
    """
    A metatransformer object that can be either a `HyperTransformer` from RDT or a `BaseSingleTableSynthesizer` from SDV.

    Args:
        data: The input data as a pandas DataFrame.
        metadata: A dictionary containing the metadata for the input data. Each key corresponds to a column name in the
            input data, and its value is a dictionary containing the metadata for the corresponding column. The metadata
            should contain "dtype" and "sdtype" fields specifying the column's data type, and a "transformer" field specifying the
            name of the transformer to use for the column and its configuration (to be instantiated below).
        dtypes: A dictionary mapping column names to their data types.
        sdv_workflow: A boolean flag indicating whether to use the SDV workflow or the RDT workflow. If True, the
            TVAESynthesizer from SDV will be used as the metatransformer. If False, the HyperTransformer from
            RDT will be used instead.
        allow_null_transformers: A boolean flag indicating whether to allow transformers to be None. If True, a None value
            for a transformer in the metadata will be treated as a valid value, and no transformer will be instantiated
            for that column.
        Synthesizer: The synthesizer class to use for the SDV workflow.
    """

    def __init__(self, metadata, sdv_workflow, allow_null_transformers, synthesizer) -> None:

        self.sdv_workflow = sdv_workflow
        self.allow_null_transformers = allow_null_transformers
        self.Synthesizer = SDV_SYNTHESIZER_CHOICES[synthesizer]
        self.dtypes = {cn: cd.get("dtype", {}) for cn, cd in metadata.items()}
        self.sdtypes = {cn: filter_dict(cd, {"dtype", "transformer"}) for cn, cd in metadata.items()}
        self.transformers = {cn: get_transformer(cd) for cn, cd in metadata.items()}

    def apply_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `data` and infers missing dtypes by reading pandas defaults.

        Returns:
            The data with the data types applied.
        """
        if not all(self.dtypes.values()):
            warnings.warn(
                f"Incomplete metadata, detecting missing `dtype`s for column(s): {[k for k, v in self.dtypes.items() if not v]} automatically...",
                UserWarning,
            )
            self.dtypes.update({cn: data[cn].dtype for cn, cv in self.dtypes.items() if not cv})
        return data.astype(self.dtypes)

    def instantiate_synthesizer(self, data: pd.DataFrame) -> BaseSingleTableSynthesizer:
        """
        Instantiates a BaseSingleTableSynthesizer object from the given metadata and data.

        Args:
            sdtypes: A dictionary of column names to their metadata, containing the key "sdtype" which
                specifies the semantic SDV data type of the column.
            transformers: A dictionary of column names to their transformers.
            data: The input DataFrame.
            allow_null_transformers: A flag indicating whether or not to allow null transformers.

        Returns:
            A BaseSingleTableSynthesizer object instantiated from the given metadata and data.
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

    def instantiate_hypertransformer(self, data: pd.DataFrame) -> HyperTransformer:
        """
        Instantiates a HyperTransformer object from the given metadata and data.

        Args:
            sdtypes: A dictionary of column names to their metadata, containing the key "sdtype" which
                specifies the semantic SDV data type of the column.
            transformers: A dictionary of column names to their transformers.
            data: The input DataFrame.
            allow_null_transformers: A flag indicating whether or not to allow null transformers.

        Returns:
            A HyperTransformer object instantiated from the given metadata and data.
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
        """Calls the appropriate instantiation method based on the value of `sdv_workflow`."""
        if self.sdv_workflow:
            return self.instantiate_synthesizer(data)
        else:
            return self.instantiate_hypertransformer(data)

    def get_dtype(self, cn: str) -> str | np.dtype:
        """Returns the dtype for the given column name `cn`."""
        return self.dtypes[cn].name if not isinstance(self.dtypes[cn], str) else self.dtypes[cn]

    def assemble(self) -> None:
        """
        Extracts the metadata for the transformers and sdtypes used to transform the data.

        Returns:
            dict[str, Any]: A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - sdtype: The data type for the column (only present if `sdv_workflow` is False).
                - transformer: A dictionary containing information about the transformer
                used for the column (if any). The dictionary has the following keys:
                - name: The name of the transformer.
                - Any other properties of the transformer that are not private or set by a random seed.
                - dtype: The data type for the column
        """
        if self.sdv_workflow:
            sdmetadata = self.metatransformer.metadata
            transformers = self.metatransformer.get_transformers()
            return {
                cn: {
                    **cd,
                    "transformer": make_transformer_dict(transformers[cn]),
                    "dtype": self.get_dtype(cn),
                }
                for cn, cd in sdmetadata.columns.items()
            }
        else:
            config = self.metatransformer.get_config()
            return {
                cn: {
                    "sdtype": cd,
                    "transformer": make_transformer_dict(config["transformers"][cn]),
                    "dtype": self.get_dtype(cn),
                }
                for cn, cd in config["sdtypes"].items()
            }

    def infer_categorical_and_continuous(self, data: pd.DataFrame) -> tuple[dict[str, int], int]:
        """
        Infers the categorical columns from the data.

        Args:
            data: The data to infer the categorical columns from.

        Returns:
            A dictionary mapping column names to the number of unique values in the column (if the column is categorical).
        """
        categoricals = {
            cn: data[cn].nunique() for cn, cd in self.assembled_metadata.items() if cd["sdtype"] == "categorical"
        }
        num_continuous = len(data.columns) - len(categoricals)
        return categoricals, num_continuous

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data by processing via the metatransformer.

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
        self.categoricals, self.num_continuous = self.infer_categorical_and_continuous(typed_data)
        prepared_data = self.prepare(typed_data)
        return prepared_data

    def get_assembled_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Returns the assembled metadata for the transformer.

        Returns:
            A dictionary mapping column names to column metadata.
                The metadata for each column has the following keys:
                - sdtype: The data type for the column (only present if `sdv_workflow` is False).
                - transformer: A dictionary containing information about the transformer
                used for the column (if any). The dictionary has the following keys:
                - name: The name of the transformer.
                - Any other properties of the transformer that are not private or set by a random seed.
                - dtype: The data type for the column

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
            A tuple containing the ordered data, a list of the number of unique values for each categorical column,
            and the number of continuous columns.

        Raises:
            ValueError: If the categorical and continuous metadata has not yet been inferred.
            ValueError: If a column is not found in the passed data.
        """
        if not self.categoricals or not self.num_continuous:
            raise ValueError(
                "Categorical and continuous metadata has not yet been inferred. Call `MetaTransformer.apply` (or `MetaTransformer.infer_categorical_and_continuous`) first."
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
