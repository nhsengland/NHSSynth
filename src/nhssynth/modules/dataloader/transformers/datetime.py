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
        Encode datetimes by converting to numeric *days* since epoch, then delegate
        to the continuous mixture transformer. Also caches train-window bounds and a
        small reservoir for later repair/jitter.
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        # Keep original name for revert
        self.original_column_name = data.name

        # Coerce to datetime64[ns] and to *float* nanoseconds (NaN for missing)
        dt = pd.to_datetime(data, errors="coerce")
        ns = dt.view("int64").astype("float64")
        ns[dt.isna()] = np.nan

        # Convert to DAYS (keeps σ in a reasonable range for mixture)
        NS_PER_DAY = float(24 * 60 * 60 * 1e9)
        self._unit_scale = NS_PER_DAY  # used later in revert
        days = ns / NS_PER_DAY

        # ---- Cache train-window bounds & a small pool (in *days*) ----
        # Work only on non-missing rows for bounds
        days_train = days[missingness_column == 0] if missingness_column is not None else days
        vals = pd.to_numeric(days_train, errors="coerce").to_numpy(dtype="float64", copy=False)
        vals = vals[np.isfinite(vals)]

        if vals.size >= 10:
            p1 = float(np.nanpercentile(vals, 1))
            p99 = float(np.nanpercentile(vals, 99))
            window = p99 - p1

            # Cache bounds in *days*
            self._days_min = p1 if window > 0 else None
            self._days_max = p99 if window > 0 else None

            # Small reservoir (in-bounds, in *days*) for repair/jitter
            pool = vals[(vals >= p1) & (vals <= p99)]
            if pool.size > 5000:
                rng = np.random.default_rng(0)
                pool = rng.choice(pool, size=5000, replace=False)
            self._days_pool = pool.astype("float64", copy=False)

            # Enable clamp only for sensible windows (>= 30 days)
            self._days_clamp_enabled = bool(window >= 30.0)

            # Debug
            p1_ts = pd.to_datetime(int(round(p1 * NS_PER_DAY)), unit="ns", errors="coerce")
            p99_ts = pd.to_datetime(int(round(p99 * NS_PER_DAY)), unit="ns", errors="coerce")
            tqdm.write(
                f"[datetime.apply] {self.original_column_name}: p1={p1_ts}  "
                f"p99={p99_ts}  Δ≈{window:.1f} days  "
                f"clamp={'ON' if self._days_clamp_enabled else 'OFF'}  pool={self._days_pool.size}"
            )
        else:
            self._days_min = self._days_max = None
            self._days_pool = np.array([], dtype="float64")
            self._days_clamp_enabled = False
            tqdm.write(f"[datetime.apply] {self.original_column_name}: insufficient data; clamp OFF, empty pool")

        # Hand off DAYS series to the mixture/continuous transformer
        days_series = pd.Series(days, index=data.index, name=self.original_column_name)
        return super().apply(
            data=days_series,
            constraint_adherence=constraint_adherence,
            missingness_column=missingness_column,
        )

    def revert(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Decode from mixture space back to *days*, clamp/jitter in *days* (if enabled),
        convert to ns and then to datetime64[ns]. Respects <base>_missing when present.
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        # 1) Let the continuous transformer put the numeric DAYS back
        reverted = super().revert(data, **kwargs)

        base = getattr(self, "original_column_name", None) or getattr(self, "name", None)
        if not base or base not in reverted.columns:
            return reverted

        # 2) DAYS as float
        days = pd.to_numeric(reverted[base], errors="coerce").astype("float64")

        # 3) Optional clamp+tiny jitter in *days*
        # --- replace the hard clip with pool-based resample + light jitter ---
        ns_min = getattr(self, "_ns_min", None)
        ns_max = getattr(self, "_ns_max", None)
        clamp_ok = bool(getattr(self, "_ns_clamp_enabled", False)) and (
            ns_min is not None and ns_max is not None and ns_max > ns_min
        )

        if clamp_ok:
            from tqdm import tqdm
            rng = np.random.default_rng()

            # out-of-bounds mask
            bad = np.isfinite(ns_vals) & ((ns_vals < ns_min) | (ns_vals > ns_max))
            tqdm.write(f"[datetime.revert] window={pd.to_datetime(ns_min)}..{pd.to_datetime(ns_max)} "
                    f"oob={int(bad.sum())}/{len(ns_vals)}")

            if bad.any():
                pool = getattr(self, "_ns_pool", None)
                if isinstance(pool, np.ndarray) and pool.size:
                    choice = rng.choice(pool, size=int(bad.sum()), replace=True).astype("float64")
                else:
                    choice = rng.uniform(ns_min, ns_max, size=int(bad.sum()))
                # small jitter so they don’t quantize to identical ticks
                J = int(max(1e8, 0.0005 * (ns_max - ns_min)))  # ≥0.1s or 0.05% window
                choice = choice + rng.integers(-J, J + 1, size=choice.shape)
                ns_vals[bad] = choice

            # light jitter for in-bounds too, to avoid edge pile-ups
            good = np.isfinite(ns_vals) & ~bad
            if good.any():
                J_small = int(max(1e7, 0.0002 * (ns_max - ns_min)))  # ≥0.01s or 0.02% window
                ns_vals[good] = ns_vals[good] + rng.integers(-J_small, J_small + 1, size=int(good.sum()))


        # 4) DAYS -> ns (float), safe cast to nullable Int64
        NS_PER_DAY = float(getattr(self, "_unit_scale", 24 * 60 * 60 * 1e9))
        ns_float = days * NS_PER_DAY
        finite = np.isfinite(ns_float)

        # Prevent int64 overflow before rounding
        i64_min, i64_max = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        ns_float = np.where(finite, np.clip(ns_float, i64_min, i64_max), np.nan)
        ns_round = np.where(finite, np.rint(ns_float), np.nan)

        # Nullable Int64 via pd.array (NaN -> <NA>)
        ns_int = pd.array(ns_round, dtype="Int64")

        # 5) To datetime64[ns]
        dt = pd.to_datetime(ns_int, unit="ns", errors="coerce")
        dt = pd.Series(dt, index=reverted.index, name=base)

        # 6) Respect missingness flag if present
        miss_col = f"{base}_missing"
        if miss_col in reverted.columns:
            m = pd.to_numeric(reverted[miss_col], errors="coerce").fillna(0).astype(bool).to_numpy()
            if m.any():
                dt.loc[m] = pd.NaT

        # 7) Write back; continuous.revert likely already dropped helper cols
        reverted[base] = dt
        return reverted
