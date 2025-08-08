import pathlib
import sys
from typing import Any, Iterable, Optional, Dict, Self, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.missingness import MISSINGNESS_STRATEGIES


class MetaTransformer:
    """
    The metatransformer is responsible for transforming input dataset into a format that can be used by the `model` module, and for transforming
    this module's output back to the original format of the input dataset.

    Args:
        dataset: The raw input DataFrame.
        metadata: Optionally, a [`MetaData`][nhssynth.modules.dataloader.metadata.MetaData] object containing the metadata for the dataset. If this is not provided it will be inferred from the dataset.
        missingness_strategy: The missingness strategy to use. Defaults to augmenting missing values in the data, see [the missingness strategies][nhssynth.modules.dataloader.missingness] for more information.
        impute_value: Only used when `missingness_strategy` is set to 'impute'. The value to use when imputing missing values in the data.

    After calling `MetaTransformer.apply()`, the following attributes and methods will be available:

    Attributes:
        typed_dataset (pd.DataFrame): The dataset with the dtypes applied.
        post_missingness_strategy_dataset (pd.DataFrame): The dataset with the missingness strategies applied.
        transformed_dataset (pd.DataFrame): The transformed dataset.
        single_column_indices (list[int]): The indices of the columns that were transformed into a single column.
        multi_column_indices (list[list[int]]): The indices of the columns that were transformed into multiple columns.

    **Methods:**

    - `get_typed_dataset()`: Returns the typed dataset.
    - `get_prepared_dataset()`: Returns the dataset with the missingness strategies applied.
    - `get_transformed_dataset()`: Returns the transformed dataset.
    - `get_multi_and_single_column_indices()`: Returns the indices of the columns that were transformed into one or multiple column(s).
    - `get_sdv_metadata()`: Returns the metadata in the correct format for SDMetrics.
    - `save_metadata()`: Saves the metadata to a file.
    - `save_constraint_graphs()`: Saves the constraint graphs to a file.

    Note that `mt.apply` is a helper function that runs `mt.apply_dtypes`, `mt.apply_missingness_strategy` and `mt.transform` in sequence.
    This is the recommended way to use the MetaTransformer to ensure that it is fully instantiated for use downstream.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        metadata: Optional[MetaData] = None,
        missingness_strategy: Optional[str] = "augment",
        impute_value: Optional[Any] = None,
    ):
        self._raw_dataset: pd.DataFrame = dataset
        self._metadata: MetaData = metadata or MetaData(dataset)
        if missingness_strategy == "impute":
            assert (
                impute_value is not None
            ), "`impute_value` of the `MetaTransformer` must be specified (via the --impute flag) when using the imputation missingness strategy"
            self._impute_value = impute_value
        self._missingness_strategy = MISSINGNESS_STRATEGIES[missingness_strategy]

    @classmethod
    def from_path(cls, dataset: pd.DataFrame, metadata_path: str, **kwargs) -> Self:
        """
        Instantiates a MetaTransformer from a metadata file via a provided path.

        Args:
            dataset: The raw input DataFrame.
            metadata_path: The path to the metadata file.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData.from_path(dataset, metadata_path), **kwargs)

    @classmethod
    def from_dict(cls, dataset: pd.DataFrame, metadata: dict, **kwargs) -> Self:
        """
        Instantiates a MetaTransformer from a metadata dictionary.

        Args:
            dataset: The raw input DataFrame.
            metadata: A dictionary of raw metadata.

        Returns:
            A MetaTransformer object.
        """
        return cls(dataset, MetaData(dataset, metadata), **kwargs)

    def drop_columns(self) -> None:
        """
        Drops columns from the dataset that are not in the `MetaData`.
        """
        self._raw_dataset = self._raw_dataset[self._metadata.columns]

    def _apply_rounding_scheme(self, working_column: pd.Series, rounding_scheme: float) -> pd.Series:
        """
        A rounding scheme takes the form of the smallest value that should be rounded to 0, i.e. 0.01 for 2dp.
        We first round to the nearest multiple in the standard way, through dividing, rounding and then multiplying.
        However, this can lead to floating point errors, so we then round to the number of decimal places required by the rounding scheme.

        e.g. `np.round(0.15 / 0.1) * 0.1` will erroneously return 0.1.

        Args:
            working_column: The column to apply the rounding scheme to.
            rounding_scheme: The rounding scheme to apply.

        Returns:
            The column with the rounding scheme applied.
        """
        working_column = np.round(working_column / rounding_scheme) * rounding_scheme
        return working_column.round(max(0, int(np.ceil(np.log10(1 / rounding_scheme)))))

    def _apply_dtype(
        self,
        working_column: pd.Series,
        column_metadata: MetaData.ColumnMetaData,
    ) -> pd.Series:
        """
        Given a `working_column`, the dtype specified in the `column_metadata` is applied to it.
         - Datetime columns are floored, and their format is inferred.
         - Rounding schemes are applied to numeric columns if specified.
         - Columns with missing values have their dtype converted to the pandas equivalent to allow for NA values.

        Args:
            working_column: The column to apply the dtype to.
            column_metadata: The metadata for the column.

        Returns:
            The column with the dtype applied.
        """
        dtype = column_metadata.dtype
        try:
            if dtype.kind == "M":
                working_column = pd.to_datetime(
                    working_column, format=column_metadata.datetime_config.get("format"), errors="coerce"
                )
                if column_metadata.datetime_config.get("floor"):
                    working_column = working_column.dt.floor(column_metadata.datetime_config.get("floor"))
                    column_metadata.datetime_config["format"] = column_metadata._infer_datetime_format(working_column)
                return working_column
            else:
                if hasattr(column_metadata, "rounding_scheme") and column_metadata.rounding_scheme is not None:
                    working_column = self._apply_rounding_scheme(working_column, column_metadata.rounding_scheme)
                # If there are missing values in the column, we need to use the pandas equivalent of the dtype to allow for NA values
                if working_column.isnull().any() and dtype.kind in ["i", "u", "f"]:
                    return working_column.astype(dtype.name.capitalize())
                else:
                    return working_column.astype(dtype)
        except ValueError:
            raise ValueError(f"{sys.exc_info()[1]}\nError applying dtype '{dtype}' to column '{working_column.name}'")

    def apply_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dtypes from the metadata to `dataset`.

        Returns:
            The dataset with the dtypes applied.
        """
        working_data = data.copy()
        for column_metadata in self._metadata:
            working_data[column_metadata.name] = self._apply_dtype(working_data[column_metadata.name], column_metadata)
        return working_data

    def apply_missingness_strategy(self) -> pd.DataFrame:
        """
        Resolves missingness in the dataset via the `MetaTransformer`'s global missingness strategy or
        column-wise missingness strategies. In the case of the `AugmentMissingnessStrategy`, the missingness
        is not resolved, instead a new column / value is added for later transformation.

        Returns:
            The dataset with the missingness strategies applied.
        """
        working_data = self.typed_dataset.copy()
        for column_metadata in self._metadata:
            if not column_metadata.missingness_strategy:
                column_metadata.missingness_strategy = (
                    self._missingness_strategy(self._impute_value)
                    if hasattr(self, "_impute_value")
                    else self._missingness_strategy()
                )
            if not working_data[column_metadata.name].isnull().any():
                continue
            working_data = column_metadata.missingness_strategy.remove(working_data, column_metadata)
        return working_data

    def apply_constraints(self) -> pd.DataFrame:
        working_data = self.post_missingness_strategy_dataset.copy()
        for constraint in self._metadata.constraints:
            working_data = constraint.transform(working_data)
        return working_data

    def _get_missingness_carrier(self, column_metadata: MetaData.ColumnMetaData) -> Union[pd.Series, Any]:
        """
        In the case of the `AugmentMissingnessStrategy`, a `missingness_carrier` has been determined for each column.
        For continuous columns this is an indicator column for the presence of NaN values.
        For categorical columns this is the value to be used to represent missingness as a category.

        Args:
            column_metadata: The metadata for the column.

        Returns:
            The missingness carrier for the column.
        """
        missingness_carrier = getattr(column_metadata.missingness_strategy, "missingness_carrier", None)
        if missingness_carrier in self.post_missingness_strategy_dataset.columns:
            return self.post_missingness_strategy_dataset[missingness_carrier]
        else:
            return missingness_carrier

    def _get_adherence_constraint(self, df) -> Union[pd.Series, Any]:

        adherence_columns = [col for col in df.columns if col.endswith("_adherence")]
        constraint_adherence = df[adherence_columns].prod(axis=1).astype(int)

        return constraint_adherence

    def transform(self) -> pd.DataFrame:
        """
        Prepares the dataset by applying each of the columns' transformers and recording the indices of the single and multi columns.

        Returns:
            The transformed dataset.
        """
        transformed_columns = []
        self.single_column_indices = []
        self.multi_column_indices = []
        col_counter = 0
        working_data = self.constrained_dataset.copy()

        # iteratively build the transformed df
        for column_metadata in tqdm(
            self._metadata, desc="Transforming data", unit="column", total=len(self._metadata.columns)
        ):
            missingness_carrier = self._get_missingness_carrier(column_metadata)
            constraint_adherence = self._get_adherence_constraint(working_data)
            transformed_data = column_metadata.transformer.apply(
                working_data[column_metadata.name], constraint_adherence, missingness_carrier
            )
            transformed_columns.append(transformed_data)

            # track single and multi column indices to supply to the model
            if isinstance(transformed_data, pd.DataFrame) and transformed_data.shape[1] > 1:
                num_to_add = transformed_data.shape[1]
                if not column_metadata.categorical:
                    self.single_column_indices.append(col_counter)
                    col_counter += 1
                    num_to_add -= 1
                self.multi_column_indices.append(list(range(col_counter, col_counter + num_to_add)))
                col_counter += num_to_add
            else:
                self.single_column_indices.append(col_counter)
                col_counter += 1

        return pd.concat(transformed_columns, axis=1)

    def apply(self) -> pd.DataFrame:
        """
        Applies the various steps of the MetaTransformer to a passed DataFrame.

        Returns:
            The transformed dataset.
        """
        self.drop_columns()
        self.typed_dataset = self.apply_dtypes(self._raw_dataset)
        self.post_missingness_strategy_dataset = self.apply_missingness_strategy()
        self.constrained_dataset = self.apply_constraints()
        self.transformed_dataset = self.transform()
        return self.transformed_dataset

    def inverse_apply(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the transformation applied by the MetaTransformer.

        Args:
            dataset: The transformed dataset.

        Returns:
            The original dataset.
        """
        for column_metadata in self._metadata:
            dataset = column_metadata.transformer.revert(dataset)
        # Enforce constraints on decoded data if available
        dataset = self.repair_constraints(dataset)

        return self.apply_dtypes(dataset)

    def get_typed_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "typed_dataset"):
            raise ValueError(
                "The typed dataset has not yet been created. Call `mt.apply()` (or `mt.apply_dtypes()`) first."
            )
        return self.typed_dataset

    def get_prepared_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "prepared_dataset"):
            raise ValueError(
                "The prepared dataset has not yet been created. Call `mt.apply()` (or `mt.apply_missingness_strategy()`) first."
            )
        return self.prepared_dataset

    def get_transformed_dataset(self) -> pd.DataFrame:
        if not hasattr(self, "transformed_dataset"):
            raise ValueError(
                "The prepared dataset has not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.transformed_dataset

    def get_multi_and_single_column_indices(self) -> tuple[list[int], list[int]]:
        """
        Returns the indices of the columns that were transformed into one or multiple column(s).

        Returns:
            A tuple containing the indices of the single and multi columns.
        """
        if not hasattr(self, "multi_column_indices") or not hasattr(self, "single_column_indices"):
            raise ValueError(
                "The single and multi column indices have not yet been created. Call `mt.apply()` (or `mt.transform()`) first."
            )
        return self.multi_column_indices, self.single_column_indices

    def get_sdv_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Calls the `MetaData` method to reformat its contents into the correct format for use with SDMetrics.

        Returns:
            The metadata in the correct format for SDMetrics.
        """
        return self._metadata.get_sdv_metadata()

    def save_metadata(self, path: pathlib.Path, collapse_yaml: bool = False) -> None:
        return self._metadata.save(path, collapse_yaml)

    def save_constraint_graphs(self, path: pathlib.Path) -> None:
        return self._metadata.constraints._output_graphs_html(path)
    
    def repair_constraints(self, df: pd.DataFrame, *, constraints=None, n_retries: int = 0) -> pd.DataFrame:
        """
        Enforce constraints on a *decoded* dataframe.

        Supported:
        - Unary bounds:  min <= col <= max         (clip)
        - Binary rels:   A <= B, <, >=, >, =       (project)
        - Sum rels:      A+B+... <= C / >= C       (proportional scale)
        - Categorical:   col in {allowed}          (snap to mode)
        """

        if constraints is None:
            constraints = getattr(self._metadata, "constraints", None)

        if constraints is None:
            return df
        
        # If it's a graph-like wrapper, try to get the iterable out
        # (be tolerant to different shapes)
        if hasattr(constraints, "constraints"):
            iterable = constraints.constraints
        elif hasattr(constraints, "__iter__"):
            iterable = constraints
        elif hasattr(constraints, "to_list"):
            iterable = constraints.to_list()
        else:
            # Nothing usable
            return df

        repaired = df.copy()

        def _clip_bounds(col: str, lo: Optional[float], hi: Optional[float]):
            if col in repaired.columns:
                s = repaired[col].to_numpy()
                if lo is not None:
                    s = np.maximum(s, lo)
                if hi is not None:
                    s = np.minimum(s, hi)
                repaired[col] = s
                
        def _as_number(x):
            try:
                if x is None: return None
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    return float(x)
                # if it's "180" in YAML, it may arrive as str
                return float(str(x))
            except Exception:
                return None

        def _ensure_binary(lhs: str, op: str, rhs: str):
            """Enforce lhs <op> rhs where rhs may be a column name OR a numeric constant."""
            if lhs not in repaired.columns:
                return
            a = repaired[lhs].to_numpy(dtype=float)

            # Decide: rhs column vs constant
            rhs_is_col = isinstance(rhs, str) and rhs in repaired.columns
            rhs_num = None if rhs_is_col else _as_number(rhs)

            if rhs_is_col:
                b = repaired[rhs].to_numpy(dtype=float)
            elif rhs_num is not None:
                b = np.full_like(a, rhs_num, dtype=float)
            else:
                # Unknown RHS; nothing to do
                return

            if op in ("<=", "<"):
                mask = a > b if op == "<=" else a >= b
                eps = 1e-8 if op == "<" else 0.0
                a[mask] = b[mask] - eps
            elif op in (">=", ">"):
                mask = a < b if op == ">=" else a <= b
                eps = 1e-8 if op == ">" else 0.0
                a[mask] = b[mask] + eps
            elif op in ("=", "=="):
                a = b
            repaired[lhs] = a

        def _ensure_sum(lhs_cols, op: str, rhs):
            """Enforce sum(lhs_cols) <op> rhs; rhs may be a column or a numeric constant."""
            cols = [c for c in lhs_cols if c in repaired.columns]
            if not cols:
                return
            L = repaired[cols].to_numpy(dtype=float)

            rhs_is_col = isinstance(rhs, str) and rhs in repaired.columns
            rhs_num = None if rhs_is_col else _as_number(rhs)
            if rhs_is_col:
                R = repaired[rhs].to_numpy(dtype=float)
            elif rhs_num is not None:
                R = np.full(L.shape[0], rhs_num, dtype=float)
            else:
                return

            s = L.sum(axis=1)
            if op in ("<=", "<"):
                mask = s > R if op == "<=" else s >= R
                scale = np.ones_like(s)
                scale[mask] = np.where(s[mask] > 0, R[mask] / s[mask], 0.0)
                L[mask] = (L[mask].T * scale[mask]).T
            elif op in (">=", ">"):
                mask = s < R if op == ">=" else s <= R
                scale = np.ones_like(s)
                scale[mask] = np.where(s[mask] != 0, R[mask] / s[mask], 1.0)
                L[mask] = (L[mask].T * scale[mask]).T

            repaired[cols] = L

        def _snap_categorical(col: str, allowed: Iterable[Any]):
            if col not in repaired.columns:
                return
            allowed = list(allowed)
            s = repaired[col].astype(object)
            mask = ~s.isin(allowed)
            if mask.any():
                mode_vals = s[s.isin(allowed)].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if len(mode_vals) else (allowed[0] if allowed else None)
                if fill_val is not None:
                    s.loc[mask] = fill_val
                    repaired[col] = s

        for c in iterable:
            cdict = c.to_dict() if hasattr(c, "to_dict") else {k: getattr(c, k) for k in dir(c) if not k.startswith("_")}
            ctype = cdict.get("type") or cdict.get("kind")
            keys = set(cdict)

            # unary bounds
            if ctype in ("UnaryBounds","Bounds","Range") or {"column","min","max"} <= keys:
                _clip_bounds(cdict.get("column"), cdict.get("min"), cdict.get("max")); continue

            # binary rel (supports RHS column OR constant)
            if ctype in ("BinaryRelation","BinaryRel") or {"lhs","op","rhs"} <= keys:
                _ensure_binary(cdict["lhs"], cdict["op"], cdict["rhs"]); continue

            # sum rel (RHS column OR constant)
            if ctype in ("SumRelation","SumRel") or {"lhs_cols","op","rhs"} <= keys:
                _ensure_sum(cdict.get("lhs_cols") or cdict.get("lhs") or [], cdict["op"], cdict["rhs"]); continue

            # categorical inclusion
            if ctype in ("InSet","CategoryInclusion") or {"column","allowed"} <= keys:
                _snap_categorical(cdict["column"], cdict["allowed"]); continue

        return repaired
