import warnings
from typing import Optional

import pandas as pd
from nhssynth.utils import filter_dict
from rdt import HyperTransformer
from rdt.transformers import *
from sdv.metadata import SingleTableMetadata
from sdv.single_table.base import BaseSingleTableSynthesizer


def get_transformer(d: dict) -> Optional[BaseSingleTableSynthesizer]:
    """
    Return a callable transformer object extracted from the given dictionary.

    Args:
        d: A dictionary containing the transformer data.

    Returns:
        A callable object (transformer) if the dictionary contains transformer data, else None.
    """
    transformer_data = d.get("transformer", None)
    if isinstance(transformer_data, dict):
        # Need to copy in case dicts are shared across columns, this can happen when reading a yaml with anchors
        transformer_data = transformer_data.copy()
        transformer_name = transformer_data.pop("name", None)
        transformer = eval(transformer_name)(**transformer_data) if transformer_name else None
    else:
        transformer = eval(transformer_data)() if transformer_data else None
    return transformer


def instantiate_synthesizer(
    sdtypes: dict[str, dict],
    transformers: dict[str, Optional[BaseTransformer]],
    data: pd.DataFrame,
    allow_null_transformers: bool,
    Synthesizer: type[BaseSingleTableSynthesizer],
) -> BaseSingleTableSynthesizer:
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
    if all(sdtypes.values()):
        metadata = SingleTableMetadata.load_from_dict({"columns": sdtypes})
    else:
        warnings.warn(
            f"Incomplete metadata, detecting missing `sdtype`s for column(s): {[k for k, v in sdtypes.items() if not v]} automatically...",
            UserWarning,
        )
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        for column_name, values in sdtypes.items():
            if values:
                metadata.update_column(column_name=column_name, **values)
    if not all(transformers.values()) and not allow_null_transformers:
        warnings.warn(
            f"Incomplete metadata, detecting missing `transformers`s for column(s): {[k for k, v in transformers.items() if not v]} automatically...",
            UserWarning,
        )
    synthesizer = Synthesizer(metadata)
    synthesizer.auto_assign_transformers(data)
    synthesizer.update_transformers(
        transformers if allow_null_transformers else {k: v for k, v in transformers.items() if v}
    )
    return synthesizer


def instantiate_hypertransformer(
    sdtypes: dict[str, dict],
    transformers: dict[str, Optional[BaseTransformer]],
    data: pd.DataFrame,
    allow_null_transformers: bool,
) -> HyperTransformer:
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
    if all(sdtypes.values()) and (all(transformers.values()) or allow_null_transformers):
        ht.set_config(
            config={
                "sdtypes": {k: v["sdtype"] for k, v in sdtypes.items()},
                "transformers": transformers,
            }
        )
    else:
        warnings.warn(
            f"Incomplete metadata, detecting missing{(' `sdtype`s for column(s): ' + str([k for k, v in sdtypes.items() if not v])) if not all(sdtypes.values()) else ''}{(' `transformer`s for column(s): ' + str([k for k, v in transformers.items() if not v])) if not all(transformers.values()) and not allow_null_transformers else ''} automatically...",
            UserWarning,
        )
        ht.detect_initial_config(data)
        ht.update_sdtypes({k: v["sdtype"] for k, v in sdtypes.items() if v})
        ht.update_transformers(
            transformers if allow_null_transformers else {k: v for k, v in transformers.items() if v}
        )
    return ht


def instantiate_metatransformer(
    metadata: dict[str, dict],
    data: pd.DataFrame,
    sdv_workflow: bool,
    allow_null_transformers: bool,
    Synthesizer: type[BaseSingleTableSynthesizer],
) -> BaseSingleTableSynthesizer | HyperTransformer:
    """
    Instantiates a metatransformer based on the given metadata and input data.

    Args:
        metadata: A dictionary containing the metadata for the input data. Each key corresponds to a column name in the
            input data, and its value is a dictionary containing the metadata for the corresponding column. The metadata
            should contain "dtype" and "sdtype" fields specifying the column's data type, and a "transformer" field specifying the
            name of the transformer to use for the column and its configuration (to be instantiated below).
        data: The input data as a pandas DataFrame.
        sdv_workflow: A boolean flag indicating whether to use the SDV workflow or the RDT workflow. If True, the
            TVAESynthesizer from SDV will be used as the metatransformer. If False, the HyperTransformer from
            RDT will be used instead.
        allow_null_transformers: A boolean flag indicating whether to allow transformers to be None. If True, a None value
            for a transformer in the metadata will be treated as a valid value, and no transformer will be instantiated
            for that column.

    Returns:
        A metatransformer object that can be either a `HyperTransformer` from RDT or a `BaseSingleTableSynthesizer` from SDV.
    """
    sdtypes = {cn: filter_dict(cd, {"dtype", "transformer"}) for cn, cd in metadata.items()}
    transformers = {cn: get_transformer(cd) for cn, cd in metadata.items()}
    if sdv_workflow:
        metatransformer = instantiate_synthesizer(sdtypes, transformers, data, allow_null_transformers, Synthesizer)
    else:
        metatransformer = instantiate_hypertransformer(sdtypes, transformers, data, allow_null_transformers)
    return metatransformer


def apply_transformer(
    metatransformer: BaseSingleTableSynthesizer | HyperTransformer,
    typed_input: pd.DataFrame,
    sdv_workflow: bool,
) -> pd.DataFrame:
    """
    Applies a metatransformer to the typed input data.

    Args:
        metatransformer: A metatransformer object that can be either a `HyperTransformer` from RDT or a `BaseSingleTableSynthesizer` from SDV.
        typed_input: The typed input data.
        sdv_workflow: A boolean flag indicating whether to use the `preprocess()` method of the `metatransformer` if it's an `SDV` synthesizer, or the `fit_transform()` method if it's an `RDT` transformer.

    Returns:
        The transformed data.
    """
    if sdv_workflow:
        transformed_input = metatransformer.preprocess(typed_input)
    else:
        transformed_input = metatransformer.fit_transform(typed_input)
    return transformed_input
