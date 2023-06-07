import pathlib
import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from nhssynth.common.dicts import filter_dict, get_key_by_value
from nhssynth.modules.dataloader.transformers import *
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


class ColumnMetaData:
    def __init__(self, name: str, data: pd.Series, raw: dict, global_missingness: bool) -> None:
        self.name = name
        print(raw["dtype"])
        self.dtype = self._validate_dtype(raw.get("dtype"), data)
        if self.dtype.kind == "M":
            print(raw["dtype"])
            self.datetime_config = self._setup_datetime_config(raw.get("dtype"), data)
        elif self.dtype.kind == "f":
            self.rounding_scheme = self._validate_rounding_scheme(raw.get("dtype"), data)
        self.categorical = self._validate_categorical(raw.get("categorical"), data)
        self.transformer = self._validate_transformer(raw.get("transformer"), data, global_missingness)
        # self.constraints = self._validate_constraints(raw.get("constraints"), data)

    def _validate_dtype(self, dtype: Optional[Union[dict, str]], data: pd.Series) -> np.dtype:
        if isinstance(dtype, dict):
            dtype_name = dtype.pop("name", None)
        elif isinstance(dtype, str):
            dtype_name = dtype
        else:
            return self._infer_dtype(data)
        try:
            return np.dtype(dtype_name)
        except TypeError:
            warnings.warn(
                f"Invalid dtype specification '{dtype_name}' for column '{self.name}', ignoring dtype for this column"
            )
            return self._infer_dtype(data)

    def _infer_dtype(self, data: pd.Series) -> np.dtype:
        return data.dtype

    def _setup_datetime_config(self, datetime_config: dict, data: pd.Series) -> dict:
        """
        Add keys to `datetime_config` corresponding to args from the `pd.to_datetime` function
        (see [the docs](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html))
        """
        if not isinstance(datetime_config, dict):
            datetime_config = {}
        else:
            datetime_config = filter_dict(datetime_config, {"name"})
        if "format" not in datetime_config:
            datetime_config["format"] = _guess_datetime_format_for_array(data[data.notna()].astype(str).to_numpy())
        if "utc" not in self.datetime_config:
            datetime_config["utc"] = data.dt.tz is not None
        return datetime_config

    def _validate_rounding_scheme(self, dtype_dict: dict, data: pd.Series) -> int:
        if dtype_dict and "rounding_scheme" in dtype_dict:
            return dtype_dict["rounding_scheme"]
        else:
            roundable_data = data[data.notna()]
            for i in range(np.finfo(roundable_data.dtype).precision):
                if (roundable_data.round(i) == roundable_data).all():
                    return i
        return None

    def _validate_categorical(self, categorical: Optional[bool], data: pd.Series) -> bool:
        if not isinstance(categorical, bool):
            warnings.warn(
                f"Invalid categorical '{categorical}' for column '{self.name}', ignoring categorical for this column"
            )
            return self._infer_categorical(data)
        else:
            return categorical

    def _infer_categorical(self, data: pd.Series) -> bool:
        return data.nunique() <= 10 or self.dtype.kind == "O"

    def _validate_transformer(
        self, transformer: Optional[Union[dict, str]], global_missingness: bool
    ) -> tuple[str, dict]:
        if isinstance(transformer, dict):
            transformer_name = transformer.pop("name", None)
            transformer_config = transformer
            if global_missingness:
                transformer_config.pop("missingness")
            else:
                missingness_strategy = transformer_config.pop("missingness", None)

        elif isinstance(transformer, str):
            transformer_name, transformer_config = transformer, {}
        else:
            return self._infer_transformer(transformer)
        try:
            return eval(transformer_name)(**transformer_config)
        except:
            warnings.warn(
                f"Invalid transformer '{transformer_name}' or transformer config for column '{self.name}', ignoring transformer for this column"
            )
            return self._infer_transformer(transformer)

    def _infer_transformer(self, transformer_config) -> GenericTransformer:
        if not isinstance(transformer_config, dict):
            transformer_config = {}
        if self.categorical:
            transformer = OHETransformer(**transformer_config)
        else:
            transformer = ClusterTransformer(**transformer_config)
        if self.dtype.kind == "M":
            transformer = DatetimeTransformer(transformer, self.datetime_config)
        return transformer


class MetaData:
    def __init__(self, data: pd.DataFrame, metadata: Optional[dict] = {}):
        self.columns: list[str] = data.columns
        self.raw_metadata: dict = metadata
        if set(self.raw_metadata.keys()) - set(self.columns):
            raise ValueError("Metadata contains keys that do not appear amongst the columns.")
        self._metadata = {cn: ColumnMetaData(cn, data[cn], self.raw_metadata.get(cn, {})) for cn in self.columns}

    def __index__(self, key: str) -> dict[str, Any]:
        return self._metadata[key]

    @classmethod
    def load(cls, path: str, data: pd.DataFrame):
        if path.exists():
            with open(path) as stream:
                metadata = yaml.safe_load(stream)
            # Filter out expanded alias/anchor groups
            metadata = filter_dict(metadata, {"transformers", "column_types"})
        else:
            warnings.warn(f"No metadata found at {path}...")
            metadata = {}
        return cls(data, metadata)

    def save(
        self,
        path: pathlib.Path,
        collapse_yaml: bool,
    ) -> None:
        """
        Writes metadata to a YAML file.

        Args:
            path: The path at which to write the metadata YAML file.
            metadata: The metadata dictionary to be written.
            collapse_yaml: A boolean indicating whether to collapse the YAML representation of the metadata, reducing duplication.
        """
        with open(path, "w") as yaml_file:
            yaml.safe_dump(
                collapse(self._metadata) if collapse_yaml else self._metadata,
                yaml_file,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_sdtypes(self) -> dict[str, str]:
        return {cn: cm.dtype.name for cn, cm in self._metadata.items()}


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
