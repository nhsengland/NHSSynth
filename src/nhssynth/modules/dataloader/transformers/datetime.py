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
        data,
        constraint_adherence=None,
        missingness_column=None,
    ):
        """
        Convert datetime column -> nanoseconds (float), cache train-range,
        then delegate to the parent transform (mixture/normalisation).
        Returns the transformed DataFrame (normalised + components [+ flags]).
        """
        import pandas as pd
        import numpy as np

        # --- normalise input to a single Series and cache names ---
        if isinstance(data, pd.DataFrame):
            if hasattr(self, "name") and self.name and self.name in data.columns:
                s = data[self.name]
            elif data.shape[1] == 1:
                s = data.iloc[:, 0]
                self.name = s.name
            else:
                raise KeyError(
                    f"[DatetimeTransformer.apply] expected column '{getattr(self,'name',None)}' in DataFrame; "
                    f"available={list(data.columns)[:10]}..."
                )
        elif isinstance(data, pd.Series):
            s = data
            if not getattr(self, "name", None):
                self.name = s.name
        else:
            raise TypeError(f"[DatetimeTransformer.apply] expected Series/DataFrame, got {type(data)}")

        self.original_column_name = s.name

        # --- to datetime, then to ns as float (so missing -> NaN) ---
        s_dt = pd.to_datetime(s, errors="coerce")
        ns = s_dt.view("int64").astype("float64")
        ns[s_dt.isna()] = np.nan

        # cache train range (unchanged)
        ns_clean = ns[~np.isnan(ns)]
        self._ns_min = int(np.nanmin(ns_clean)) if ns_clean.size else None
        self._ns_max = int(np.nanmax(ns_clean)) if ns_clean.size else None

        # 🔧 build a clean, aligned 0/1 missingness mask
        if missingness_column is None:
            miss = s_dt.isna().astype(int)
        else:
            miss = (
                pd.to_numeric(missingness_column, errors="coerce")
                .reindex(s_dt.index)
                .fillna(0)
                .astype(int)
            )
        miss.name = f"{self.original_column_name}_missing"

        ns.name = self.original_column_name
        return super().apply(
            data=ns,
            constraint_adherence=constraint_adherence,
            missingness_column=miss,
        )


    def revert(self, data, **kwargs):
        """
        Reconstruct datetime from decoded nanoseconds.
        - Accepts Series or DataFrame from super().revert
        - Clamps to training ns-range (if cached)
        - Applies missingness mask (if present)
        - Converts to datetime64[ns]
        - Drops helper columns (normalised, comps, missing/adherence)
        """
        import re
        import numpy as np
        import pandas as pd

        reverted = super().revert(data, **kwargs)

        # get the decoded ns series back
        if isinstance(reverted, pd.Series):
            series = reverted
        else:
            if self.original_column_name not in reverted.columns:
                raise KeyError(
                    f"[DatetimeTransformer.revert] expected column '{self.original_column_name}' "
                    f"in reverted DataFrame; available={list(reverted.columns)[:10]}..."
                )
            series = reverted[self.original_column_name]

        # coerce to float ns
        series = pd.to_numeric(series, errors="coerce").astype("float64")

        # clamp to training range if available
        ns_min = getattr(self, "_ns_min", None)
        ns_max = getattr(self, "_ns_max", None)
        if ns_min is not None and ns_max is not None:
            series = series.clip(lower=ns_min, upper=ns_max)

        # apply missingness if the flag exists
        miss_col = f"{self.original_column_name}_missing"
        if miss_col in data.columns:
            mask = pd.to_numeric(data[miss_col], errors="coerce").fillna(0).astype(bool).to_numpy()
            series[mask] = np.nan

        # convert back to datetime
        data[self.original_column_name] = pd.to_datetime(series, unit="ns", errors="coerce")

        # drop helper columns for this datetime feature
        base = self.original_column_name
        drop_cols = []
        if f"{base}_normalised" in data.columns:
            drop_cols.append(f"{base}_normalised")
        drop_cols += [c for c in data.columns if re.fullmatch(rf"{re.escape(base)}_c\d+", c)]
        drop_cols += [c for c in (f"{base}_missing", f"{base}_adherence") if c in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols, errors="ignore")

        return data


