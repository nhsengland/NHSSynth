import argparse
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nhssynth.common.constants import TRACKED_METRICS
from nhssynth.common.strings import add_spaces_before_caps
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Model(nn.Module, ABC):
    """
    Abstract base class for all NHSSynth models

    Args:
        data: The data to train on
        metatransformer: A `MetaTransformer` to use for converting the generated data to match the original data
        batch_size: The batch size to use during training
        use_gpu: Flag to determine whether to use the GPU (if available)

    Attributes:
        nrows: The number of rows in the `data`
        ncols: The number of columns in the `data`
        columns: The names of the columns in the `data`
        metatransformer: The `MetaTransformer` (potentially) associated with the model
        multi_column_indices: A list of lists of column indices, where each sublist containts the indices for a one-hot encoded column
        single_column_indices: Indices of all non-onehot columns
        data_loader: A PyTorch DataLoader for the `data`
        private: Whether the model is private, i.e. whether the `DPMixin` class has been inherited
        device: The device to use for training (CPU or GPU)

    Raises:
        TypeError: If the `Model` class is directly instantiated (i.e. not inherited)
        AssertionError: If the number of columns in the `data` does not match the number of indices in `multi_column_indices` and `single_column_indices`
        UserWarning: If `use_gpu` is True but no GPU is available
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metatransformer: MetaTransformer,
        batch_size: int = 32,
        use_gpu: bool = False,
    ) -> None:
        if type(self) is Model:
            raise TypeError("Cannot directly instantiate the `Model` class")
        super().__init__()

        self.nrows, self.ncols = data.shape
        self.columns: pd.Index = data.columns

        self.metatransformer = metatransformer
        self.multi_column_indices: list[list[int]] = metatransformer.multi_column_indices
        self.single_column_indices: list[int] = metatransformer.single_column_indices
        assert len(self.single_column_indices) + sum([len(x) for x in self.multi_column_indices]) == self.ncols

        self.data_loader: DataLoader = DataLoader(
            # TODO Should the data also all be turned into floats?
            TensorDataset(torch.Tensor(data.to_numpy())),
            pin_memory=True,
            batch_size=batch_size,
        )

        self.setup_device(use_gpu)

    def setup_device(self, use_gpu: bool) -> None:
        """Sets up the device to use for training (CPU or GPU) depending on `use_gpu` and device availability."""
        if use_gpu:
            if torch.cuda.is_available():
                self.device: torch.device = torch.device("cuda:0")
            else:
                warnings.warn("`use_gpu` was provided but no GPU is available, using CPU")
        self.device: torch.device = torch.device("cpu")

    def save(self, filename: str) -> None:
        """Saves the model to `filename`."""
        torch.save(self.state_dict(), filename)

    def load(self, path: str) -> None:
        """Loads the model from `path`."""
        self.load_state_dict(torch.load(path))

    @classmethod
    @abstractmethod
    def get_args() -> list[str]:
        """Returns the list of arguments to look for in an `argparse.Namespace`, these must map to the arguments of the inheritor."""
        raise NotImplementedError

    def _start_training(self, num_epochs: int, patience: int, displayed_metrics: list[str]) -> None:
        """
        Initialises the training process.

        Args:
            num_epochs: The number of epochs to train for
            patience: The number of epochs to wait before stopping training early if the loss does not improve
            displayed_metrics: The metrics to display during training, this should be set to an empty list if running `train` in a notebook or the output may be messy

        Attributes:
            metrics: A dictionary of lists of tracked metrics, where each list contains the values for each batch
            stats_bars: A dictionary of tqdm status bars for each tracked metric
            max_length: The maximum length of the tracked metric names, used for formatting the tqdm status bars
            start_time: The time at which training started
            update_time: The time at which the tqdm status bars were last updated
        """
        self.num_epochs = num_epochs
        self.patience = patience
        self.metrics = {metric: np.empty(0, dtype=float) for metric in TRACKED_METRICS}
        if not hasattr(self, "target_epsilon"):
            self.metrics.pop("Privacy", None)
            if "Privacy" in displayed_metrics:
                displayed_metrics.remove("Privacy")
        self.stats_bars = {
            metric: tqdm(total=0, desc="", position=i, bar_format="{desc}", leave=True)
            for i, metric in enumerate(displayed_metrics)
        }
        self.max_length = max([len(add_spaces_before_caps(s)) + 5 for s in displayed_metrics] + [20])
        self.start_time = self.update_time = time.time()

    def _generate_metric_str(self, key) -> str:
        """Generates a string to display the current value of the metric `key`."""
        return f"{(add_spaces_before_caps(key) + ':').ljust(self.max_length)}  {np.mean(self.metrics[key][-len(self.data_loader) :]):.4f}"

    def _record_metrics(self, losses):
        """Records the metrics for the current batch to file and updates the tqdm status bars."""
        for key in self.metrics.keys():
            if key in losses:
                if losses[key]:
                    self.metrics[key] = np.append(self.metrics[key], losses[key].item())
        if time.time() - self.update_time > 0.5:
            for key, stats_bar in self.stats_bars.items():
                stats_bar.set_description_str(self._generate_metric_str(key))
                self.update_time = time.time()

    def _check_patience(self, epoch: int, metric: float) -> bool:
        """Maintains `_min_metric` and `_stop_counter` to determine whether to stop training early according to `patience`."""
        if epoch == 0:
            self._stop_counter = 0
            self._min_metric = metric
            self._patience_delta = self._min_metric / 1e4
        if metric < (self._min_metric - self._patience_delta):
            self._min_metric = metric
            self._stop_counter = 0  # Set counter to zero
        else:  # elbo has not improved
            self._stop_counter += 1
        return self._stop_counter == self.patience

    def _finish_training(self, num_epochs: int) -> None:
        """Closes each of the tqdm status bars and prints the time taken to do `num_epochs`."""
        for stats_bar in self.stats_bars.values():
            stats_bar.close()
        tqdm.write(f"Completed {num_epochs} epochs in {time.time() - self.start_time:.2f} seconds.\033[0m")
