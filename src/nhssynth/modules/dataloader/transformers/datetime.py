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

        # Cache robust train bounds to avoid edge pileups on revert
        from tqdm import tqdm
        # Cache robust train bounds to avoid edge pileups on revert
        ns_clean = ns[~np.isnan(ns)]
        if ns_clean.size:
            q1 = np.nanpercentile(ns_clean, 1)
            q99 = np.nanpercentile(ns_clean, 99)

            from tqdm import tqdm, trange
            tqdm.write(f"[datetime.apply] bounds (p1,p99): {pd.to_datetime(int(q1))} → {pd.to_datetime(int(q99))} "
                    f"(Δ={(q99 - q1)/1e9/86400:.1f} days)")

            # enable clamp only if the window is not degenerate
            if (q99 - q1) >= (30 * 24 * 3600 * 1e9):   # ≥ 30 days in ns
                self._ns_min = int(q1)
                self._ns_max = int(q99)
                tqdm.write("[datetime.apply] clamp ENABLED")
            else:
                self._ns_min = None
                self._ns_max = None
                tqdm.write("[datetime.apply] clamp DISABLED (train window too narrow)")
        else:
            self._ns_min = self._ns_max = None
            from tqdm import tqdm
            tqdm.write("[datetime.apply] no valid values; clamp DISABLED")

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

        # Clamp to training window if available (prevents edge collapse)
        from tqdm import tqdm
        ns_min = getattr(self, "_ns_min", None)
        ns_max = getattr(self, "_ns_max", None)
        if ns_min is not None and ns_max is not None:
            width_days = (ns_max - ns_min) / (24 * 3600 * 1e9)
            from tqdm import tqdm
            tqdm.write(f"[datetime.revert] clamping into [{pd.to_datetime(ns_min)} .. {pd.to_datetime(ns_max)}] "
                    f"(Δ≈{width_days:.1f} d)")
            ns_vals = np.clip(ns_vals, ns_min, ns_max)

        # Cast to integer nanoseconds and to datetime
        # Use pandas nullable Int64 to carry NaNs through
        INT64_MIN = np.iinfo(np.int64).min
        INT64_MAX = np.iinfo(np.int64).max

        # ensure float array
        ns_vals = np.asarray(ns_vals, dtype="float64")

        # mask bad
        finite = np.isfinite(ns_vals)

        # clip into int64 range (leave non-finite as-is for now)
        ns_vals[finite] = np.clip(ns_vals[finite], INT64_MIN, INT64_MAX)

        # round to nearest int (still float dtype)
        ns_vals[finite] = np.rint(ns_vals[finite])

        # build a nullable Int64 with an explicit mask (True = missing)
        mask = ~finite
        # values must be int64 where not masked
        int_arr = np.empty_like(ns_vals, dtype="int64")
        # np.rint may have produced -9.22e18 exactly; ensure within range again
        int_arr[finite] = np.clip(ns_vals[finite], INT64_MIN, INT64_MAX).astype("int64")
        # dummy fill for masked positions (ignored by mask)
        int_arr[mask] = 0

        # construct pandas Nullable Int64
        ns_int = pd.arrays.IntegerArray(int_arr, mask)

        # now to datetime
        dt = pd.to_datetime(ns_int, unit="ns", errors="coerce")

        # Respect missingness flag if present (force NaT)
        miss_col = f"{base}_missing"
        if miss_col in reverted.columns:
            miss_mask = reverted[miss_col].astype(bool)
            dt[miss_mask] = pd.NaT

        reverted[base] = dt

        # Drop helper cols for this feature if your pipeline expects it
        # (The continuous revert may already drop *_value/_c* etc.)
        return reverted
