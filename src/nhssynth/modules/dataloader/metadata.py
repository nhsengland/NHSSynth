import pathlib
import warnings

import numpy as np
import pandas as pd
import yaml
from nhssynth.modules.dataloader.utils import *
from nhssynth.utils import get_key_by_value


def create_empty_metadata(data: pd.DataFrame) -> dict[str, dict]:
    return {cn: {} for cn in data.columns}


def check_metadata_columns(metadata: dict, data: pd.DataFrame):
    assert all([k in data.columns for k in metadata.keys()])
    metadata.update({cn: {} for cn in data.columns if cn not in metadata})


def load_metadata(in_path: pathlib.Path, data: pd.DataFrame):
    if in_path.exists():
        with open(in_path) as stream:
            metadata = yaml.safe_load(stream)
        metadata.pop("transformers", None)
        metadata.pop("column_types", None)
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


def assemble_metadata(dtypes, metatransformer, sdv_workflow):
    if sdv_workflow:
        sdmetadata, synthesizer = metatransformer
        transformers = synthesizer.get_transformers()
        metadata = {
            cn: {
                **cd,
                **{
                    "transformer": {
                        "name": type(transformers[cn]).__name__,
                        **filter_inner_dict(
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
                    **filter_inner_dict(
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


def output_metadata(out_path: pathlib.Path, dtypes, metatransformer, sdv_workflow=True, collapse_yaml=True):

    metadata = assemble_metadata(dtypes, metatransformer, sdv_workflow)

    if collapse_yaml:
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
        metadata = {"transformers": transformers, "column_types": column_types, **metadata}

    with open(out_path, "w") as yaml_file:
        yaml.safe_dump(metadata, yaml_file, default_flow_style=False, sort_keys=False)
