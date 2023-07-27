from typing import Final, Union

import numpy as np
import pandas as pd


class Constraint:
    _OPERATOR_MAP: Final = {
        "<": pd.Series.lt,
        "<=": pd.Series.le,
        ">": pd.Series.gt,
        ">=": pd.Series.ge,
    }

    def __init__(
        self,
        base: str,
        operator: str,
        reference: Union[str, float],
        range: str = None,
        reference_is_column: bool = False,
        ignore: bool = False,
    ):
        self.base = base
        self.operator = operator
        self.reference = reference
        self.reference_is_column = reference_is_column
        self.range = range
        self._ignore = ignore

    def __str__(self):
        return f"{self.base} {self.operator} {self.reference}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.base == other.base
            and self.operator == other.operator
            and self.reference == other.reference
            and self.reference_is_column == other.reference_is_column
        )

    def format(self):
        if self._ignore:
            return None
        else:
            return f"{self.base} {self.operator if not self.range else 'in'} {self.reference if not self.range else self.range}"

    def transform(self, df):
        base = df[self.base]
        if self.reference_is_column:
            reference = df[self.reference][base.index]
        else:
            reference = self.reference

        adherence = base._OPERATOR_MAP[self.operator](reference).astype(int)
        adherence[reference.isna()] = 1
        # When there is no reference, i.e. admidate and disdate, constraint is disdate >= admidate and admidate is null, we assume we want to keep the ability to generate disdates without a reference admidate, so we require a new column that inherits the constraints of the base column except for this constraint
        diff = abs(base[adherence] - reference[adherence])
        diff.fillna(diff.mean(), inplace=True)
        df[self.base + "_diff"] = np.log(diff + 1)
        df[self.base + "_adherence"] = adherence
        return df
