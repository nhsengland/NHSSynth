import numpy as np
import pandas as pd
import torch

from nhssynth.modules.model.common.model import Model


class Marginal(Model):
    """
    Zero-order baseline: samples each column independently from its empirical distribution.

    Fits on the raw post-missingness data (``metatransformer.post_missingness_strategy_dataset``)
    and generates samples in the original data space without using the metatransformer pipeline.
    By construction this baseline preserves no inter-variable correlations — any decent
    generative model should outperform it on metrics that measure joint distributions
    (CorrelationSimilarity, ContingencySimilarity, downstream tasks).
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
        self._col_values = {col: df[col].dropna().values for col in df.columns}
        self._finish_training(1)
        return 1, {}

    def generate(self, N: int = None) -> pd.DataFrame:
        N = N or self.nrows
        result = {col: np.random.choice(vals, size=N, replace=True) for col, vals in self._col_values.items()}
        return pd.DataFrame(result)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Marginal sampler has no forward pass")

    def save(self, filename: str) -> None:
        torch.save({"col_values": self._col_values}, filename)

    def load(self, path: str) -> None:
        data = torch.load(path)
        self._col_values = data["col_values"]
