import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yaml
from nhssynth.utils import filter_dict, get_key_by_value
from rdt import HyperTransformer


def create_empty_metadata(data: pd.DataFrame) -> dict[str, dict]:
    return {cn: {} for cn in data.columns}


def check_metadata_columns(metadata: dict, data: pd.DataFrame):
    assert all([k in data.columns for k in metadata.keys()])
    metadata.update({cn: {} for cn in data.columns if cn not in metadata})


def load_metadata(in_path: pathlib.Path, data: pd.DataFrame):
    if in_path.exists():
        with open(in_path) as stream:
            metadata = yaml.safe_load(stream)
        metadata = filter_dict(metadata, {"transformers", "column_types"})
    else:
        metadata = create_empty_metadata(data)
    check_metadata_columns(metadata, data)
    return metadata


def instantiate_dtypes(metadata: dict, data: pd.DataFrame) -> dict[str, np.dtype]:
    dtypes = {cn: cd.get("dtype", {}) for cn, cd in metadata.items()}
    if not all(dtypes.values()):
        warnings.warn(
            f"Incomplete metadata, detecting missing `dtype`s for column(s): {[k for k, v in dtypes.items() if not v]} automatically...",
            UserWarning,
        )
        dtypes.update({cn: data[cn].dtype for cn, cv in dtypes.items() if not cv})
    return dtypes


def assemble_metadata(
    dtypes: dict[str, type],
    metatransformer: Any | HyperTransformer,
    sdv_workflow: bool,
) -> dict[str, dict[str, Any]]:
    """
    Constructs a metadata dictionary from a list of data types and a metatransformer.

    Args:
        dtypes: A dictionary mapping column names of the input data to their assigned data types.
        metatransformer: A meta-transformer used to create synthetic data.
            If `sdv_workflow` is True, `metatransformer` should be an SDV synthesizer.
            If `sdv_workflow` is False, `metatransformer` should be a dictionary containing transformers and sdtypes for each column.
        sdv_workflow: A boolean indicating whether the data was transformed using the SDV / synthesizer workflow.

    Returns:
        dict[str, dict[str, Any]]: A dictionary mapping column names to column metadata.
            The metadata for each column has the following keys:
            - dtype: The name of the data type for the column.
            - sdtype: The data type for the column (only present if `sdv_workflow` is False).
            - transformer: A dictionary containing information about the transformer
              used for the column (if any). The dictionary has the following keys:
              - name: The name of the transformer.
              - Any other properties of the transformer that are not private or set by a random seed.
    """
    if sdv_workflow:
        sdmetadata = metatransformer.metadata
        transformers = metatransformer.get_transformers()
        metadata = {
            cn: {
                **cd,
                **{
                    "transformer": {
                        "name": type(transformers[cn]).__name__,
                        **filter_dict(
                            transformers[cn].__dict__,
                            {"output_properties", "random_states", "transform", "reverse_transform", "_dtype"},
                        ),
                    }
                    if transformers[cn]
                    else None,
                    "dtype": dtypes[cn].name if not isinstance(dtypes[cn], str) else dtypes[cn],
                },
            }
            for cn, cd in sdmetadata.columns.items()
        }
    else:
        config = metatransformer.get_config()
        metadata = {
            cn: {
                "sdtype": cd,
                "transformer": {
                    "name": type(config["transformers"][cn]).__name__,
                    **filter_dict(
                        config["transformers"][cn].__dict__,
                        {"output_properties", "random_states", "transform", "reverse_transform", "_dtype"},
                    ),
                }
                if config["transformers"][cn]
                else None,
                "dtype": dtypes[cn].name if not isinstance(dtypes[cn], str) else dtypes[cn],
            }
            for cn, cd in config["sdtypes"].items()
        }
    return metadata


def collapse(metadata: dict) -> dict:
    """
    Given a metadata dictionary, rewrites it to collapse duplicate column types and transformers.

    Args:
        metadata (dict): The metadata dictionary to be rewritten.

    Returns:
        dict: A rewritten metadata dictionary with collapsed column types and transformers.
            The returned dictionary has the following structure:
            {
                "transformers": dict,
                "column_types": dict,
                **metadata
            }
            - "transformers" is a dictionary mapping transformer indices (integers) to transformer configurations.
            - "column_types" is a dictionary mapping column type indices (integers) to column type configurations.
            - "**metadata" contains the original metadata dictionary, with column types and transformers
              rewritten to use the indices in "transformers" and "column_types".
    """
    c_index = 1
    column_types = {}
    t_index = 1
    transformers = {}
    for cn, cd in metadata.items():

        if cd not in column_types.values():
            column_types[c_index] = cd.copy()
            metadata[cn] = column_types[c_index]
            c_index += 1
        else:
            cix = get_key_by_value(column_types, cd)
            metadata[cn] = column_types[cix]

        if cd["transformer"] not in transformers.values() and cd["transformer"]:
            transformers[t_index] = cd["transformer"].copy()
            metadata[cn]["transformer"] = transformers[t_index]
            t_index += 1
        elif cd["transformer"]:
            tix = get_key_by_value(transformers, cd["transformer"])
            metadata[cn]["transformer"] = transformers[tix]

    return {"transformers": transformers, "column_types": column_types, **metadata}


def output_metadata(
    out_path: pathlib.Path,
    dtypes: dict[str, Any],
    metatransformer: Any | HyperTransformer,
    sdv_workflow: bool = True,
    collapse_yaml: bool = True,
) -> None:
    """
    Writes metadata to a YAML file.

    Args:
        out_path: The path at which to write the metadata YAML file.
        dtypes: A dictionary mapping column names of the input data to their assigned data types.
        metatransformer: The synthesizer or hypertransformer that was used to transform the data.
        sdv_workflow: A boolean indicating whether the data was transformed using the SDV / synthesizer workflow.
        collapse_yaml: A boolean indicating whether to collapse the YAML representation of the metadata, reducing duplication.

    Returns:
        None
    """
    metadata = assemble_metadata(dtypes, metatransformer, sdv_workflow)

    if collapse_yaml:
        metadata = collapse(metadata)

    with open(out_path, "w") as yaml_file:
        yaml.safe_dump(metadata, yaml_file, default_flow_style=False, sort_keys=False)
