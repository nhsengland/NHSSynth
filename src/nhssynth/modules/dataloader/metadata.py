import pathlib
from typing import Any

import pandas as pd
import yaml
from nhssynth.common.dicts import filter_dict, get_key_by_value


def create_empty_metadata(data: pd.DataFrame) -> dict[str, dict]:
    """
    Creates an empty metadata dictionary for a given pandas DataFrame.

    Args:
        data: The DataFrame in question.

    Returns:
        A dictionary where each key corresponds to a column name in the DataFrame, and each value is an empty dictionary.
    """
    return {cn: {} for cn in data.columns}


def check_metadata_columns(metadata: dict[str, dict[str, Any]], data: pd.DataFrame) -> None:
    """
    Check if all column representations in the `metadata` correspond to valid columns in the `data`.
    If any columns are not present, add them to the metadata and instantiate an empty dictionary.

    Args:
        metadata: A dictionary containing metadata for the columns in the passed `data`.
        data: The DataFrame to check against the metadata.

    Raises:
        AssertionError: If any columns that *are* in metadata are *not* present in the `data`.
    """
    assert all([k in data.columns for k in metadata.keys()])
    metadata.update({cn: {} for cn in data.columns if cn not in metadata})


def load_metadata(in_path: pathlib.Path, data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Load metadata from a YAML file located at `in_path`. If the file does not exist, create an empty metadata
    dictionary with column names from the `data`.

    Args:
        in_path: The path to the YAML file containing the metadata.
        data: The DataFrame containing the data for which metadata is being loaded.

    Returns:
        A metadata dictionary containing information about the columns in the `data`.
    """
    if in_path.exists():
        with open(in_path) as stream:
            metadata = yaml.safe_load(stream)
        # Filter out expanded alias/anchor groups
        metadata = filter_dict(metadata, {"transformers", "column_types"})
        check_metadata_columns(metadata, data)
    else:
        metadata = create_empty_metadata(data)
    return metadata


def collapse(metadata: dict) -> dict:
    """
    Given a metadata dictionary, rewrite to collapse duplicate column types and transformers in order to leverage YAML anchors

    Args:
        metadata: The metadata dictionary to be rewritten.

    Returns:
        dict: A rewritten metadata dictionary with collapsed column types and transformers.
            The returned dictionary has the following structure:
            {
                "transformers": dict,
                "column_types": dict,
                **metadata  # one entry for each column that now reference the dicts above
            }
            - "transformers" is a dictionary mapping transformer indices to transformer configurations.
            - "column_types" is a dictionary mapping column type indices to column type configurations.
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
    metadata: dict[str, dict[str, Any]],
    collapse_yaml: bool,
) -> None:
    """
    Writes metadata to a YAML file.

    Args:
        out_path: The path at which to write the metadata YAML file.
        metadata: The metadata dictionary to be written.
        collapse_yaml: A boolean indicating whether to collapse the YAML representation of the metadata, reducing duplication.
    """
    if collapse_yaml:
        metadata = collapse(metadata)
    with open(out_path, "w") as yaml_file:
        yaml.safe_dump(metadata, yaml_file, default_flow_style=False, sort_keys=False)


def get_sdtypes(metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, str]]]:
    """
    Extracts the `sdtype` for each column from a valid assembled metadata dictionary and reformats them the correct format for use with SDMetrics.

    Args:
        metadata: The metadata dictionary to extract the `sdtype`s from.

    Returns:
        A dictionary mapping column names to a dict containing `sdtype` value for that column.
    """
    return {
        "columns": {
            cn: {
                "sdtype": cd["sdtype"],
            }
            for cn, cd in metadata.items()
        }
    }
