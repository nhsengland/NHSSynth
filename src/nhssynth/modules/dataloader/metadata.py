import pathlib
import warnings
from typing import Any, Iterator, Optional, Union

import numpy as np
import pandas as pd
import yaml
from nhssynth.common.dicts import filter_dict, get_key_by_value
from nhssynth.modules.dataloader.missingness import (
    MISSINGNESS_STRATEGIES,
    GenericMissingnessStrategy,
)
from nhssynth.modules.dataloader.transformers import *
from nhssynth.modules.dataloader.transformers.generic import GenericTransformer
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


class ColumnMetaData:
    def __init__(self, name: str, data: pd.Series, raw: dict) -> None:
        self.name = name
        self.dtype: np.dtype = self._validate_dtype(data, raw.get("dtype"))
        self.categorical: bool = self._validate_categorical(data, raw.get("categorical"))
        self.missingness_strategy: GenericMissingnessStrategy = self._validate_missingness_strategy(
            raw.get("missingness")
        )
        self.transformer: GenericTransformer = self._validate_transformer(raw.get("transformer"))
        # self.constraints = self._validate_constraints(raw.get("constraints"), data)

    def _validate_dtype(self, data: pd.Series, dtype_raw: Optional[Union[dict, str]] = None) -> np.dtype:
        if isinstance(dtype_raw, dict):
            dtype_name = dtype_raw.pop("name", None)
        elif isinstance(dtype_raw, str):
            dtype_name = dtype_raw
        else:
            dtype_name = self._infer_dtype(data)
        try:
            dtype = np.dtype(dtype_name)
        except TypeError:
            warnings.warn(
                f"Invalid dtype specification '{dtype_name}' for column '{self.name}', ignoring dtype for this column"
            )
            dtype = self._infer_dtype(data)
        if dtype.kind == "M":
            self._setup_datetime_config(data, dtype_raw)
        # elif dtype.kind == "f":
        #     self.rounding_scheme = self._validate_rounding_scheme(data, dtype_raw)
        return dtype

    def _infer_dtype(self, data: pd.Series) -> np.dtype:
        return data.dtype.name

    def _setup_datetime_config(self, data: pd.Series, datetime_config: dict) -> dict:
        """
        Add keys to `datetime_config` corresponding to args from the `pd.to_datetime` function
        (see [the docs](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html))
        """
        if not isinstance(datetime_config, dict):
            datetime_config = {}
        else:
            datetime_config = filter_dict(datetime_config, {"format"}, include=True)
        if "format" not in datetime_config:
            datetime_config["format"] = _guess_datetime_format_for_array(data[data.notna()].astype(str).to_numpy())
        self.datetime_config = datetime_config

    # def _validate_rounding_scheme(self, dtype_dict: dict, data: pd.Series) -> int:
    #     if dtype_dict and "rounding_scheme" in dtype_dict:
    #         return dtype_dict["rounding_scheme"]
    #     else:
    #         roundable_data = data[data.notna()]
    #         for i in range(np.finfo(roundable_data.dtype).precision):
    #             if (roundable_data.round(i) == roundable_data).all():
    #                 return i
    #     return None

    def _validate_categorical(self, data: pd.Series, categorical: Optional[bool] = None) -> bool:
        if categorical is None:
            return self._infer_categorical(data)
        elif not isinstance(categorical, bool):
            warnings.warn(
                f"Invalid categorical '{categorical}' for column '{self.name}', ignoring categorical for this column"
            )
            return self._infer_categorical(data)
        else:
            self.boolean = data.nunique() <= 2
            return categorical

    def _infer_categorical(self, data: pd.Series) -> bool:
        self.boolean = data.nunique() <= 2
        return data.nunique() <= 10 or self.dtype.kind == "O"

    def _validate_missingness_strategy(self, missingness_strategy: Optional[Union[dict, str]]) -> tuple[str, dict]:
        if not missingness_strategy:
            return None
        if isinstance(missingness_strategy, dict):
            impute = missingness_strategy.get("impute", None)
            strategy = "impute" if impute else missingness_strategy.get("strategy", None)
        else:
            strategy = missingness_strategy
        if (
            strategy not in MISSINGNESS_STRATEGIES
            or (strategy == "impute" and impute == "mean" and self.dtype.kind != "f")
            or (strategy == "impute" and not impute)
        ):
            warnings.warn(
                f"Invalid missingness strategy '{missingness_strategy}' for column '{self.name}', ignoring missingness strategy for this column"
            )
            return None
        return MISSINGNESS_STRATEGIES[strategy](impute) if strategy == "impute" else MISSINGNESS_STRATEGIES[strategy]()

    def _validate_transformer(self, transformer: Optional[Union[dict, str]] = {}) -> tuple[str, dict]:
        # if transformer is neither a dict nor a str statement below will raise a TypeError
        if isinstance(transformer, dict):
            self.transformer_name = transformer.get("name")
            self.transformer_config = filter_dict(transformer, "name")
        elif isinstance(transformer, str):
            self.transformer_name = transformer
            self.transformer_config = {}
        else:
            if transformer is not None:
                warnings.warn(
                    f"Invalid transformer config '{transformer}' for column '{self.name}', ignoring transformer for this column"
                )
            self.transformer_name = None
            self.transformer_config = {}
        if not self.transformer_name:
            return self._infer_transformer()
        else:
            try:
                if self.transformer_name == "DatetimeTransformer":
                    return eval(self.transformer_name)(**self.transformer_config, **self.datetime_config)
                else:
                    return eval(self.transformer_name)(**self.transformer_config)
            except:
                warnings.warn(
                    f"Invalid transformer '{self.transformer_name}' or config '{self.transformer_config}' for column '{self.name}', ignoring transformer for this column"
                )
                return self._infer_transformer()

    def _infer_transformer(self) -> GenericTransformer:
        if self.categorical:
            transformer = OHETransformer(**self.transformer_config)
        else:
            transformer = ClusterTransformer(**self.transformer_config)
        if self.dtype.kind == "M":
            transformer = DatetimeTransformer(transformer, **self.datetime_config)
        return transformer


class MetaData:
    def __init__(self, data: pd.DataFrame, metadata: Optional[dict] = {}):
        self.columns: pd.Index = data.columns
        self.raw_metadata: dict = metadata
        if set(self.raw_metadata.keys()) - set(self.columns):
            raise ValueError("Metadata contains keys that do not appear amongst the columns.")
        self._metadata = {cn: ColumnMetaData(cn, data[cn], self.raw_metadata.get(cn, {})) for cn in self.columns}

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self._metadata[key]

    def __iter__(self) -> Iterator:
        return iter(self._metadata.values())

    @classmethod
    def load(cls, path_str: str, data: pd.DataFrame):
        path = pathlib.Path(path_str)
        if path.exists():
            with open(path) as stream:
                metadata = yaml.safe_load(stream)
            # Filter out expanded alias/anchor groups
            metadata = filter_dict(metadata, {"column_types"})
        else:
            warnings.warn(f"No metadata found at {path}...")
            metadata = {}
        return cls(data, metadata)

    def save(self, path: pathlib.Path, collapse_yaml: bool) -> None:
        """
        Writes metadata to a YAML file.

        Args:
            path: The path at which to write the metadata YAML file.
            metadata: The metadata dictionary to be written.
            collapse_yaml: A boolean indicating whether to collapse the YAML representation of the metadata, reducing duplication.
        """
        with open(path, "w") as yaml_file:
            yaml.safe_dump(
                self.assemble(collapse_yaml),
                yaml_file,
                default_flow_style=False,
                sort_keys=False,
            )

    def assemble(self, collapse_yaml: bool) -> dict[str, dict[str, Any]]:
        assembled_metadata = {
            cn: {
                "dtype": cmd.dtype.name
                if not hasattr(cmd, "datetime_config")
                else {"name": cmd.dtype.name, **cmd.datetime_config},
                "categorical": cmd.categorical,
            }
            for cn, cmd in self._metadata.items()
        }
        # Only add "missingness" to assembled_data if it is not None
        for cn, cmd in self._metadata.items():
            if cmd.missingness_strategy:
                assembled_metadata[cn]["missingness"] = (
                    cmd.missingness_strategy.name
                    if cmd.missingness_strategy.name != "impute"
                    else {"name": cmd.missingness_strategy.name, "impute": cmd.missingness_strategy.impute}
                )
            if cmd.transformer_config:
                assembled_metadata[cn]["transformer"] = {
                    **cmd.transformer_config,
                    "name": cmd.transformer.__class__.__name__,
                }
        if collapse_yaml:
            assembled_metadata = self.collapse(assembled_metadata)
        return assembled_metadata

    def __repr__(self) -> None:
        return yaml.dump(self._metadata, default_flow_style=False, sort_keys=False)

    def collapse(self, metadata: dict) -> dict:
        """
        Given a metadata dictionary, rewrite to collapse duplicate column types in order to leverage YAML anchors and shrink the file.

        Args:
            metadata: The metadata dictionary to be rewritten.

        Returns:
            dict: A rewritten metadata dictionary with collapsed column types and transformers.
                The returned dictionary has the following structure:
                {
                    "column_types": dict,
                    **metadata  # one entry for each column that now reference the dicts above
                }
                - "column_types" is a dictionary mapping column type indices to column type configurations.
                - "**metadata" contains the original metadata dictionary, with column types and transformers
                rewritten to use the indices in "transformers" and "column_types".
        """
        c_index = 1
        column_types = {}
        column_type_counts = {}
        for cn, cd in metadata.items():
            if cd not in column_types.values():
                column_types[c_index] = cd.copy()
                column_type_counts[c_index] = 1
                c_index += 1
            else:
                cix = get_key_by_value(column_types, cd)
                column_type_counts[cix] += 1

        for cn, cd in metadata.items():
            cix = get_key_by_value(column_types, cd)
            if column_type_counts[cix] > 1:
                metadata[cn] = column_types[cix]
            else:
                column_types.pop(cix)

        return {"column_types": {i + 1: x for i, x in enumerate(column_types.values())}, **metadata}

    def get_sdmetadata(self) -> dict[str, dict[str, dict[str, str]]]:
        sdmetadata = {
            "columns": {
                cn: {
                    "sdtype": "boolean"
                    if cmd.boolean
                    else "categorical"
                    if cmd.categorical
                    else "datetime"
                    if cmd.dtype.kind == "M"
                    else "numerical",
                }
                for cn, cmd in self._metadata.items()
            }
        }
        for cn, cmd in self._metadata.items():
            if cmd.dtype.kind == "M":
                sdmetadata["columns"][cn]["format"] = cmd.datetime_config["format"]
        return sdmetadata
