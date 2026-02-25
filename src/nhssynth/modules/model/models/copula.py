import numpy as np
import pandas as pd
import scipy.stats
import torch

from nhssynth.modules.model.common.model import Model


class Copula(Model):
    """
    Gaussian copula baseline with ordinal marginals for categorical columns.

    Fits on the raw post-missingness data (``metatransformer.post_missingness_strategy_dataset``)
    and generates samples in the original data space without using the metatransformer pipeline.

    Each column's marginal is modelled as:

    - **Categorical**: ordinal marginal — each observation is mapped to
      ``u ~ Uniform(F(x−), F(x))`` where F is the empirical CDF over categories,
      then inverted via ``np.searchsorted`` on the CDF.
    - **Datetime**: converted to ``int64`` nanoseconds and treated as continuous.
    - **Continuous**: rank-based probability integral transform,
      ``u = rank / (n + 1)``, giving an approximately uniform marginal.

    The joint structure is captured by fitting a multivariate Gaussian to the
    PIT-transformed data.  A small diagonal regularisation (``1e-6 * I``) is added
    to the estimated covariance for numerical stability.

    This baseline captures linear inter-variable correlations — more than ``Marginal``
    but less than non-linear deep generative models.
    """

    @classmethod
    def get_args(cls) -> list[str]:
        return []

    @classmethod
    def get_metrics(cls) -> list[str]:
        return []

    def train(self, num_epochs: int, patience: int, displayed_metrics: list, notebook_run: bool = False):
        self._start_training(num_epochs, patience, displayed_metrics, notebook_run)
        df = self.metatransformer.post_missingness_strategy_dataset
        self._fit(df)
        self._finish_training(1)
        return 1, {}

    def _fit(self, df: pd.DataFrame) -> None:
        meta_lookup = {m.name: m for m in self.metatransformer._metadata}
        # Drop any row with NaN in any column — the Gaussian copula requires consistent row counts
        df = df.dropna()
        self._columns = list(df.columns)
        n = len(df)
        gaussian_data = np.zeros((n, len(self._columns)))
        self._marginals: dict = {}

        for i, col in enumerate(self._columns):
            series = df[col]
            meta = meta_lookup.get(col)
            is_cat = meta.categorical if meta else False
            is_dt = pd.api.types.is_datetime64_any_dtype(series)

            if is_dt:
                vals = series.astype("int64").values
                ranks = scipy.stats.rankdata(vals)
                u = np.clip(ranks / (n + 1), 1e-6, 1 - 1e-6)
                gaussian_data[:, i] = scipy.stats.norm.ppf(u)
                self._marginals[col] = {"type": "datetime", "vals_sorted": np.sort(vals)}

            elif is_cat:
                cats, counts = np.unique(series.values, return_counts=True)
                probs = counts / counts.sum()
                cdf = np.cumsum(probs)
                cdf_lower = np.concatenate([[0.0], cdf[:-1]])
                cat_to_idx = {c: j for j, c in enumerate(cats)}
                u = np.array([np.random.uniform(cdf_lower[cat_to_idx[v]], cdf[cat_to_idx[v]]) for v in series.values])
                u = np.clip(u, 1e-6, 1 - 1e-6)
                gaussian_data[:, i] = scipy.stats.norm.ppf(u)
                self._marginals[col] = {"type": "categorical", "cats": cats, "cdf": cdf}

            else:  # continuous
                vals = series.values.astype(float)
                ranks = scipy.stats.rankdata(vals)
                u = np.clip(ranks / (n + 1), 1e-6, 1 - 1e-6)
                gaussian_data[:, i] = scipy.stats.norm.ppf(u)
                self._marginals[col] = {"type": "continuous", "vals_sorted": np.sort(vals)}

        self._gauss_mean = gaussian_data.mean(axis=0)
        cov = np.atleast_2d(np.cov(gaussian_data.T))
        self._gauss_cov = cov + 1e-6 * np.eye(len(self._columns))

    def generate(self, N: int = None) -> pd.DataFrame:
        N = N or self.nrows
        z = np.random.multivariate_normal(self._gauss_mean, self._gauss_cov, size=N)
        u = scipy.stats.norm.cdf(z)  # (N, n_cols), values in (0, 1)
        result = {}

        for i, col in enumerate(self._columns):
            marginal = self._marginals[col]
            u_col = np.clip(u[:, i], 1e-6, 1 - 1e-6)

            if marginal["type"] == "datetime":
                vals_sorted = marginal["vals_sorted"]
                idxs = np.clip((u_col * len(vals_sorted)).astype(int), 0, len(vals_sorted) - 1)
                result[col] = pd.to_datetime(vals_sorted[idxs])

            elif marginal["type"] == "categorical":
                cats, cdf = marginal["cats"], marginal["cdf"]
                idxs = np.clip(np.searchsorted(cdf, u_col, side="left"), 0, len(cats) - 1)
                result[col] = cats[idxs]

            else:  # continuous
                vals_sorted = marginal["vals_sorted"]
                idxs = np.clip((u_col * len(vals_sorted)).astype(int), 0, len(vals_sorted) - 1)
                result[col] = vals_sorted[idxs]

        return pd.DataFrame(result)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Copula has no forward pass")

    def save(self, filename: str) -> None:
        torch.save(
            {
                "gauss_mean": self._gauss_mean,
                "gauss_cov": self._gauss_cov,
                "marginals": self._marginals,
                "columns": self._columns,
            },
            filename,
        )

    def load(self, path: str) -> None:
        data = torch.load(path)
        self._gauss_mean = data["gauss_mean"]
        self._gauss_cov = data["gauss_cov"]
        self._marginals = data["marginals"]
        self._columns = data["columns"]
