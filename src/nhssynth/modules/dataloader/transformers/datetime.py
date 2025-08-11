from typing import Optional

import numpy as np
import pandas as pd

from nhssynth.modules.dataloader.transformers.base import (
    ColumnTransformer,
    TransformerWrapper,
)


class DatetimeTransformer(TransformerWrapper):
    """
    A transformer to convert datetime features to numeric features. Before applying an underlying (wrapped) transformer.
    The datetime features are converted to nanoseconds since the epoch, and missing values are assigned to 0.0 under the `AugmentMissingnessStrategy`.

    Args:
        transformer: The [`ColumnTransformer`][nhssynth.modules.dataloader.transformers.base.ColumnTransformer] to wrap.

    After applying the transformer, the following attributes will be populated:

    Attributes:
        original_column_name: The name of the original column.
    """

    def __init__(self, transformer: ColumnTransformer) -> None:
        super().__init__(transformer)

    def apply(
        self,
        data: pd.Series,
        constraint_adherence: Optional[pd.Series],
        missingness_column: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Encode datetimes by first converting to a numeric scale (days since epoch),
        then delegating to the continuous mixture transformer. This keeps mixture σ
        in a sane range (days), avoiding huge nanosecond-scale variances.
        """
        # Keep original name for revert
        self.original_column_name = data.name

        # Coerce to datetime
        dt = pd.to_datetime(data, errors="coerce")

        # Convert to int64 nanoseconds (float for NaN handling)
        ns = dt.view("int64").astype("float64")
        ns[dt.isna()] = np.nan

        # in src/nhssynth/modules/dataloader/transformers/datetime.py
        # inside DatetimeTransformer.apply(), after you compute `ns` and before returning

        from tqdm import tqdm

        # Work on non-missing training values
        ns_train = ns[missingness_column == 0] if missingness_column is not None else ns
        ns_vals = ns_train.to_numpy(dtype="float64", copy=False)
        ns_vals = ns_vals[np.isfinite(ns_vals)]

        if ns_vals.size >= 10:
            q1  = float(np.nanpercentile(ns_vals, 1))
            q99 = float(np.nanpercentile(ns_vals, 99))
            window = q99 - q1
            # cache bounds & a small empirical pool for later resampling
            self._ns_min = int(q1) if window > 0 else None
            self._ns_max = int(q99) if window > 0 else None
            # keep a small reservoir of in-bounds training samples for DOB repair/jitter
            # (store as int64 nanoseconds)
            pool = ns_vals[(ns_vals >= q1) & (ns_vals <= q99)]
            if pool.size > 5000:
                rng = np.random.default_rng(0)
                pool = rng.choice(pool, size=5000, replace=False)
            self._ns_pool = pool.astype("int64", copy=False)
            # enable clamp only if the window is sensible (>= 30 days)
            min_ns_window = 30 * 24 * 3600 * 1e9
            self._ns_clamp_enabled = bool(window >= min_ns_window)
            tqdm.write(
                f"[datetime.apply] p1={pd.to_datetime(int(q1))} p99={pd.to_datetime(int(q99))} "
                f"Δ≈{window/1e9/86400:.1f}d clamp={'ON' if self._ns_clamp_enabled else 'OFF'} "
                f"(pool={len(self._ns_pool)})"
            )
        else:
            self._ns_min = self._ns_max = None
            self._ns_pool = np.array([], dtype="int64")
            self._ns_clamp_enabled = False
            tqdm.write("[datetime.apply] insufficient data for bounds; clamp OFF, empty pool")


        # Work in DAYS to keep σ reasonable
        NS_PER_DAY = float(24 * 60 * 60 * 1e9)
        self._unit_scale = NS_PER_DAY  # used in revert
        days = ns / NS_PER_DAY

        # Name + index preserved for downstream
        days_series = pd.Series(days, index=data.index, name=self.original_column_name)

        # Delegate to the wrapped continuous transformer (mixture/GMM)
        return super().apply(
            data=days_series,
            constraint_adherence=constraint_adherence,
            missingness_column=missingness_column,
        )

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Decode from mixture space back to numeric DAYS, then scale to ns and cast
        to datetime64[ns], with robust train-window clamping. Preserves missingness.
        """
        # First, let the continuous transformer decode back to numeric DAYS
        reverted = super().revert(data, **kwargs)

        base = self.original_column_name
        if base not in reverted.columns:
            # Nothing to do
            return reverted

        # Numeric days -> ns -> datetime64[ns]
        series = pd.to_numeric(reverted[base], errors="coerce").astype("float64")

        NS_PER_DAY = float(getattr(self, "_unit_scale", 24 * 60 * 60 * 1e9))
        ns_vals = series * NS_PER_DAY  # STILL float here for NaN safety

        # Optional clamp
        ns_min = getattr(self, "_ns_min", None)
        ns_max = getattr(self, "_ns_max", None)
        clamp_ok = bool(getattr(self, "_ns_clamp_enabled", False)) and (ns_min is not None) and (ns_max is not None) and (ns_max > ns_min)

        from tqdm import tqdm
        if clamp_ok:
            tqdm.write(f"[datetime.revert] clamp to {pd.to_datetime(ns_min)} .. {pd.to_datetime(ns_max)}")
            ns_vals = np.clip(ns_vals, ns_min, ns_max)

            # add a tiny in-window jitter so values don't all quantize to the same tick
            # jitter amplitude ~ 0.25% of the window, but at least 1 second
            J = int(max(1e9, 0.0025 * (ns_max - ns_min)))
            rng = np.random.default_rng()
            ns_vals = ns_vals + rng.integers(-J, J + 1, size=ns_vals.shape)

        # Safe cast to nullable Int64 (unchanged from our hardened block)
        # Safe cast to nullable Int64 and to datetime Series aligned to `reverted`
        INT64_MIN = np.iinfo(np.int64).min
        INT64_MAX = np.iinfo(np.int64).max

        finite = np.isfinite(ns_vals)
        ns_vals = np.where(finite, np.clip(ns_vals, INT64_MIN, INT64_MAX), ns_vals)
        ns_vals = np.where(finite, np.rint(ns_vals), ns_vals)

        int_arr = np.zeros_like(ns_vals, dtype="int64")
        int_arr[finite] = np.clip(ns_vals[finite], INT64_MIN, INT64_MAX).astype("int64")
        mask = np.isnan(ns_vals)  # boolean numpy array

        ns_int = pd.arrays.IntegerArray(int_arr, mask)

        # IMPORTANT: build a Series so we can assign by mask later
        dt = pd.to_datetime(
            pd.Series(ns_int, index=reverted.index, name=base),
            unit="ns",
            errors="coerce",
        )

        # Respect missingness if present (mask must be a boolean ndarray)
        miss_col = f"{base}_missing"
        if miss_col in data.columns:
            m = data[miss_col].to_numpy(dtype=bool, copy=False)
            # Series supports boolean assignment; Index would not
            dt.loc[m] = pd.NaT

        # Write back into the decoded DataFrame and return it
        reverted[base] = dt
        return reverted

