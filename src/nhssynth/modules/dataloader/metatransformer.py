import pathlib
import sys
from typing import Any, Iterable, Optional, Dict, Self, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from tqdm import tqdm
import inspect

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
                    working_column,
                    format=column_metadata.datetime_config.get("format"),
                    errors="coerce",
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

    def _call_transformer_apply(
        self,
        transformer,
        *,
        series,
        constraint_adherence=None,
        missingness_column=None,
    ):
        """
        Call transformer.apply with only the kwargs it supports, using **keywords only**.
        - Binds the input series to 'data' if present, otherwise to the first non-self parameter.
        - Only passes constraint_adherence / missingness_column if the transformer declares them.
        """
        fn = transformer.apply
        sig = inspect.signature(fn)
        params = sig.parameters

        # Decide which parameter name to bind the series to
        if "data" in params:
            data_param = "data"
        else:
            data_param = next((n for n, p in params.items() if n != "self"), None)
            if data_param is None:
                # Last resort: the method takes no args beyond self; try calling without any
                return fn()

        kwargs = {data_param: series}

        if "constraint_adherence" in params:
            kwargs["constraint_adherence"] = constraint_adherence
        if "missingness_column" in params:
            kwargs["missingness_column"] = missingness_column

        return fn(**kwargs)


    def transform(self) -> pd.DataFrame:
        """
        Apply each column transformer to its *raw* Series, then concatenate results.
        Ensures each transformer receives a single Series (not the whole/mutated DataFrame),
        which fixes DateTime ('dob') KeyErrors.
        """

        # Prefer the dataset that already has missingness flags computed,
        # but still contains the original raw columns.
        if hasattr(self, "post_missingness_strategy_dataset") and self.post_missingness_strategy_dataset is not None:
            source_df = self.post_missingness_strategy_dataset
        elif hasattr(self, "typed_dataset") and self.typed_dataset is not None:
            source_df = self.typed_dataset
        else:
            # Fallback: raw dataset as last resort
            source_df = self._raw_dataset

        parts = []

        # Helper: get a column if it exists, else None
        def _maybe_col(df: pd.DataFrame, name: str):
            return df[name] if (df is not None and name in df.columns) else None

        for col_meta in self._metadata:
            # Work out the original column name this transformer handles
            col = (
                getattr(col_meta, "name", None)
                or getattr(col_meta, "column", None)
                or getattr(col_meta, "feature", None)
            )
            if col is None:
                raise ValueError(f"Metadata entry missing column name: {col_meta}")

            # Always hand the transformer a *Series* from the original (pre-transform) frame
            if col not in source_df.columns:
                raise KeyError(
                    f"[MetaTransformer.transform] Expected raw column '{col}' in source_df; "
                    f"available={list(source_df.columns)[:15]}..."
                )
            series = source_df[col]

            # Optional per-row flags
            # Missingness: prefer exact "{col}_missing"; if not present, try to find any "<col>_missing*"
            miss = _maybe_col(source_df, f"{col}_missing")
            if miss is None:
                # Try a looser match if you have variant names
                candidates = [c for c in source_df.columns if c.startswith(f"{col}_missing")]
                miss = source_df[candidates[0]] if candidates else None

            # Constraint adherence: if your PR1 kept these in constrained_dataset, pass it through; else use ones
            if hasattr(self, "constrained_dataset") and self.constrained_dataset is not None:
                adh = _maybe_col(self.constrained_dataset, f"{col}_adherence")
            else:
                adh = None

            if adh is None:
                # Default to all-ones (i.e., include all rows during transform)
                adh = pd.Series(1, index=series.index, name=f"{col}_adherence", dtype=int)

            # Apply the per-column transformer
            part = self._call_transformer_apply(
                col_meta.transformer,
                series=series,
                constraint_adherence=adh,
                missingness_column=miss,
            )

            # Normalise to DataFrame
            if isinstance(part, pd.Series):
                part = part.to_frame()

            value_idx_in_part = None
            for cand in (f"{col}_value", f"{col}_normalized", f"{col}_normalised"):
                if cand in part.columns:
                    value_idx_in_part = part.columns.get_loc(cand)
                    break

            # record absolute index for this value column
            if not hasattr(self, "continuous_value_indices"):
                self.continuous_value_indices = []

            abs_offset = sum(m.shape[1] if hasattr(m, "shape") else 1 for m in parts)  # cols before this part
            if value_idx_in_part is not None:
                self.continuous_value_indices.append(abs_offset + value_idx_in_part)

            parts.append(part)

        # Concatenate all transformed parts
        transformed = pd.concat(parts, axis=1)
        
        multi_groups: list[list[int]] = []
        single_list: list[int] = []

        col_offset = 0
        for part in parts:
            m = part.shape[1] if hasattr(part, "shape") else 1
            if m > 1:
                multi_groups.append(list(range(col_offset, col_offset + m)))
            else:
                single_list.append(col_offset)
            col_offset += m

        # Store on self for the model
        self.multi_column_indices = multi_groups
        self.single_column_indices = single_list
        self.output_columns = list(transformed.columns)
        self.ncols = transformed.shape[1]
        self.continuous_value_indices = list(self.continuous_value_indices)

        # Make sure downstream code (like VAE.generate) has the correct column names
        self.columns = list(transformed.columns)

        return transformed


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
        
        def _zstat(df, base):
            for sfx in ("_value","_normalized","_normalised"):
                col = f"{base}{sfx}"
                if col in df.columns:
                    v = df[col].to_numpy()
                    tqdm.write(f"[pre-revert df] {col}: std={float(np.nanstd(v)):.4f}, "
                            f"min={np.nanmin(v):.4f}, max={np.nanmax(v):.4f}")
                    return
            tqdm.write(f"[pre-revert df] {base}: NO VALUE COLUMN FOUND")

        _zstat(dataset, "x8")
        _zstat(dataset, "dob")
        
        # binarize generated missingness indicators: >0.5 -> 1, else 0
        for col in list(dataset.columns):
            if col.endswith("_missing"):
                v = pd.to_numeric(dataset[col], errors="coerce").fillna(0.0).to_numpy()
                dataset[col] = (v > 0.5).astype(int)

        
        for column_metadata in self._metadata:
            dataset = column_metadata.transformer.revert(dataset)
        
        # --- DEBUG with tqdm.write ---
        def _dbg(tag):
            try:
                msgs = []
                if "x8" in dataset:
                    msgs.append(
                        f"x8 uniques={dataset['x8'].nunique(dropna=False)} "
                        f"min/max={dataset['x8'].min()} / {dataset['x8'].max()}"
                    )
                if "dob" in dataset:
                    msgs.append(
                        f"dob min/max={dataset['dob'].min()} / {dataset['dob'].max()}"
                    )
                tqdm.write(f"[{tag}] " + " | ".join(msgs))
            except Exception as e:
                tqdm.write(f"[{tag}] debug failed: {e}")

        _dbg("post-revert")
        
        # Ensure the dataset has the same columns as the original
        # Enforce constraints on decoded data if available
        dataset = self.repair_constraints(dataset, mode="reflect")

        _dbg("post-constraints")

        out = self.apply_dtypes(dataset)
        _dbg("post-dtypes") 
        return out

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

    def repair_constraints(self, df: pd.DataFrame, *, mode="reflect", rng=None, n_retries: int = 0) -> pd.DataFrame:
        """
        Enforce constraints on a *decoded* DataFrame.

        Reads constraints from self._metadata.constraints.minimal_constraints.

        Supports:
        - Numeric bounds via <, <=, >, >=, =  (column vs. constant or column)
        - Categorical membership via 'in'
        """
    
        if rng is None:
            rng = np.random.default_rng()    
    
        constraints = getattr(getattr(self, "_metadata", None), "constraints", None)
        if constraints is None:
            return df
        constraints_iterable = getattr(constraints, "minimal_constraints", None)
        if constraints_iterable is None:
            return df

        repaired = df.copy()

        # ----------------- helpers -----------------
        def _as_number(x):
            try:
                if x is None:
                    return None
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    return float(x)
                return float(str(x))
            except Exception:
                return None

        def _norm_op(op):
            if op is None:
                return None
            op = str(op).strip().lower()
            return {
                "<": "<",
                "lt": "<",
                "<=": "<=",
                "lte": "<=",
                ">": ">",
                "gt": ">",
                ">=": ">=",
                "gte": ">=",
                "=": "=",
                "==": "=",
                "eq": "=",
                "in": "in",
            }.get(op, op)

        def _ensure_binary_or_in(base: str, op: str, reference, reference_is_column: bool):
            if base not in repaired.columns:
                return

            if op == "in":
                # categorical inclusion
                if reference_is_column:
                    if isinstance(reference, str) and reference in repaired.columns:
                        allowed = set(repaired[reference].dropna().unique().tolist())
                    else:
                        return
                else:
                    if isinstance(reference, (list, tuple, set)):
                        allowed = set(reference)
                    else:
                        allowed = set(str(reference).split(","))
                s = repaired[base].astype(object)
                mask = ~s.isin(allowed)
                if mask.any():
                    mode_vals = s[s.isin(allowed)].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if len(mode_vals) else next(iter(allowed))
                    s.loc[mask] = fill_val
                    repaired[base] = s
                return

            if op == "=":
                if reference_is_column and isinstance(reference, str) and reference in repaired.columns:
                    repaired[base] = repaired[reference]
                else:
                    ref_num = _as_number(reference)
                    if ref_num is not None:
                        repaired[base] = ref_num
                    else:
                        # equality to a non-numeric constant → treat as categorical fill
                        repaired[base] = reference
                return

            # numeric comparisons
            a = repaired[base].to_numpy(dtype=float)
            if reference_is_column:
                if not (isinstance(reference, str) and reference in repaired.columns):
                    return
                b = repaired[reference].to_numpy(dtype=float)
            else:
                ref_num = _as_number(reference)
                if ref_num is None:
                    return
                b = np.full_like(a, ref_num, dtype=float)

            if op in ("<", "<="):
                mask = a >= b if op == "<" else a > b
                if not mask.any():
                    repaired[base] = a; return
                if mode == "clip":
                    eps = 1e-8 if op == "<" else 0.0
                    a[mask] = b[mask] - eps
                elif mode == "reflect":
                    # reflect over the boundary then clip just in case
                    a[mask] = 2.0 * b[mask] - a[mask]
                    # ensure strictness
                    eps = 1e-8 if op == "<" else 0.0
                    a[mask] = np.minimum(a[mask], b[mask] - eps)
                elif mode == "uniform":
                    # sample uniformly just below the bound within window w
                    w = np.clip(0.05 * np.nanstd(a), 1e-6, np.inf)  # 5% of std as window
                    eps = 1e-8 if op == "<" else 0.0
                    a[mask] = rng.uniform(b[mask] - w, b[mask] - eps)
                elif mode == "resample":
                    # bootstrap from in-bounds upper tail (e.g., above q=0.95)
                    inb = a[~mask]
                    if inb.size:
                        q = np.quantile(inb, 0.95)
                        tail = inb[inb >= q]
                        if tail.size == 0:
                            tail = inb
                        a[mask] = rng.choice(tail, size=mask.sum(), replace=True)
                    else:
                        # fallback to uniform if everything violated
                        eps = 1e-8 if op == "<" else 0.0
                        a[mask] = b[mask] - eps
                repaired[base] = a
            elif op in (">", ">="):
                mask = a <= b if op == ">" else a < b
                if not mask.any():
                    repaired[base] = a; return
                if mode == "clip":
                    eps = 1e-8 if op == ">" else 0.0
                    a[mask] = b[mask] + eps
                elif mode == "reflect":
                    a[mask] = 2.0 * b[mask] - a[mask]
                    eps = 1e-8 if op == ">" else 0.0
                    a[mask] = np.maximum(a[mask], b[mask] + eps)
                elif mode == "uniform":
                    w = np.clip(0.05 * np.nanstd(a), 1e-6, np.inf)
                    eps = 1e-8 if op == ">" else 0.0
                    a[mask] = rng.uniform(b[mask] + eps, b[mask] + w)
                elif mode == "resample":
                    inb = a[~mask]
                    if inb.size:
                        q = np.quantile(inb, 0.05)
                        tail = inb[inb <= q]
                        if tail.size == 0:
                            tail = inb
                        a[mask] = rng.choice(tail, size=mask.sum(), replace=True)
                    else:
                        eps = 1e-8 if op == ">" else 0.0
                        a[mask] = b[mask] + eps
                repaired[base] = a
                
        def _get_pool_for(self, base: str):
            """
            Return a numeric pool of in-bounds training values for `base`:
            - For datetime columns: int64 nanoseconds (as float for NaN handling)
            - For numeric columns: float64
            Falls back to raw/typed dataset if transformer pool absent.
            """
            # 1) Try a transformer-provided reservoir (e.g., DatetimeTransformer._ns_pool)
            try:
                t = next(m.transformer for m in self._metadata if getattr(m, "name", None) == base)
                pool = getattr(t, "_ns_pool", None)
                if isinstance(pool, np.ndarray) and pool.size:
                    # convert to float for np.isfinite downstream
                    return pool.astype("float64", copy=False)
            except StopIteration:
                pass

            # 2) Fall back to a source dataset column
            src = getattr(self, "typed_dataset", None)
            if src is None:
                src = getattr(self, "_raw_dataset", None)
            if src is None or base not in src.columns:
                return None

            col = src[base]

            # Datetime -> int64 ns -> float64
            if is_datetime64_any_dtype(col):
                arr = pd.to_datetime(col, errors="coerce").view("int64").astype("float64")
            else:
                # Numeric or other -> force numeric (non-convertible -> NaN)
                arr = pd.to_numeric(col, errors="coerce").astype("float64")

            # Keep only finite values
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None

            # Downsample very large pools
            if arr.size > 10000:
                rng = np.random.default_rng(0)
                arr = rng.choice(arr, size=10000, replace=False)

            return arr
                
        def _fold_into_interval(a: np.ndarray, low: float, high: float,
                        strict_low: bool, strict_high: bool) -> np.ndarray:
            """
            Reflect-then-wrap any values into [low, high] (or (low, high), etc.).
            Works in one shot, no ping-pong. Vectorised.
            """
            a = a.astype(float, copy=False)
            w = high - low
            if not np.isfinite(w) or w <= 0:
                return a  # degenerate interval, nothing to do

            # Reflect-wrap into [0, 2w), then fold to [0, w]
            y = (a - low) % (2 * w)
            y = np.where(y <= w, y, 2 * w - y)
            out = low + y

            # handle strict bounds by nudging inside by tiny eps
            eps = 1e-8
            if strict_low:
                out = np.where(out <= low, low + eps, out)
            else:
                out = np.where(out < low, low, out)
            if strict_high:
                out = np.where(out >= high, high - eps, out)
            else:
                out = np.where(out > high, high, out)
            return out
        
        def _bootstrap_into_interval(a: np.ndarray, low: float, high: float, rng) -> np.ndarray:
            """Resample any out-of-range values from the in-bounds portion of `a`.
            If there are no in-bounds values, fall back to uniform(low, high)."""
            a = a.astype(float, copy=False)
            bad = (a < low) | (a > high)
            if not np.any(bad):
                return a
            inb = a[~bad]
            if inb.size > 0:
                a[bad] = rng.choice(inb, size=bad.sum(), replace=True)
            else:
                a[bad] = rng.uniform(low, high, size=bad.sum())
            return a


        # ----------------- apply constraints -----------------
        from tqdm import tqdm

        # ----------------- apply constraints (iteratively) -----------------
        max_passes = max(1, int(n_retries) + 2)  # 2 passes by default
        total_changes = 0

        for pass_idx in range(max_passes):
            pass_changes = 0

            for c in constraints_iterable:
                cd = c.to_dict() if hasattr(c, "to_dict") else {
                    k: getattr(c, k) for k in dir(c) if not k.startswith("_")
                }
                base = cd.get("base")
                op = _norm_op(cd.get("operator"))
                ref = cd.get("reference")
                ref_is_col = bool(cd.get("reference_is_column", False))

                if not base or base not in repaired.columns or not op:
                    continue

                # ----- snapshot BEFORE on the *repaired* DataFrame -----
                before = repaired[base].copy()

                # apply into 'repaired'
                _ensure_binary_or_in(base, op, ref, ref_is_col)

                # ----- AFTER & deltas -----
                after = repaired[base]
                # Count true numeric changes (ignore NaN==NaN)
                changed_mask = ~(before.eq(after) | (before.isna() & after.isna()))
                n_changed = int(changed_mask.sum())
                pass_changes += n_changed
                total_changes += n_changed

                # targeted debug for x8
                if base == "x8":
                    tqdm.write(f"[repair] pass {pass_idx+1}: x8 adjusted {n_changed} rows for {op} {ref}")

            # Early exit if nothing else changed on this pass
            if pass_changes == 0:
                break
            
            # Collect lower/upper constant bounds per base
            bounds = {}
            for c in constraints_iterable:
                cd  = c.to_dict() if hasattr(c, "to_dict") else {k: getattr(c, k) for k in dir(c) if not k.startswith("_")}
                base = cd.get("base"); op = _norm_op(cd.get("operator")); ref = cd.get("reference")
                if not base or base not in repaired.columns or not op:
                    continue
                if bool(cd.get("reference_is_column", False)):
                    continue
                ref_num = _as_number(ref)
                if ref_num is None:
                    continue

                b = bounds.setdefault(base, {"strict_low": False, "strict_high": False})
                if op in (">", ">="):
                    b["low"]  = max(ref_num, b.get("low", -np.inf)) if "low" in b else ref_num
                    b["strict_low"]  = b["strict_low"] or (op == ">")
                elif op in ("<", "<="):
                    b["high"] = min(ref_num, b.get("high",  np.inf)) if "high" in b else ref_num
                    b["strict_high"] = b["strict_high"] or (op == "<")

            # after you assembled `bounds` dict {base: {'low':..,'high':..}}
            changes_interval = 0
            for base, b in bounds.items():
                low, high = b.get("low", -np.inf), b.get("high", np.inf)
                if not (np.isfinite(low) and np.isfinite(high) and high > low):
                    continue

                a = repaired[base].to_numpy(dtype=float, copy=False)
                bad = (a < low) | (a > high)
                if not bad.any():
                    continue

                if mode == "fold":
                    a = _fold_into_interval(a, low, high, b["strict_low"], b["strict_high"])
                elif mode == "clip":
                    eps_lo = 1e-8 if b["strict_low"] else 0.0
                    eps_hi = 1e-8 if b["strict_high"] else 0.0
                    a = np.where(a < low + eps_lo, low + eps_lo, a)
                    a = np.where(a > high - eps_hi, high - eps_hi, a)
                elif mode == "uniform":
                    a[bad] = rng.uniform(low, high, size=bad.sum())
                elif mode == "resample":
                    pool = _get_pool_for(base)
                    if pool is not None:
                        inb = pool[(pool >= low) & (pool <= high)]
                        if inb.size:
                            a[bad] = rng.choice(inb, size=bad.sum(), replace=True)
                        else:
                            a[bad] = rng.uniform(low, high, size=bad.sum())
                    else:
                        a[bad] = rng.uniform(low, high, size=bad.sum())
                else:
                    # default: reflect (kept for backward compatibility)
                    a = _fold_into_interval(a, low, high, b["strict_low"], b["strict_high"])

                repaired[base] = a
                changes_interval += int(bad.sum())
                if base == "x8":
                    tqdm.write(f"[repair:{mode}] x8 adjusted {int(bad.sum())} into [{low},{high}]")
            tqdm.write(f"[repair] interval fix-ups: {changes_interval}")

        return repaired

