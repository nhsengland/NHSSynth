import re

import numpy as np
import torch


class CTGANConditionalSampler:
    """
    Manages conditional vector construction and conditioned data sampling for CTGAN.

    At each training step CTGAN:
    1. Randomly selects one categorical column.
    2. Samples a category from that column's empirical distribution.
    3. Builds a one-hot condition vector spanning all categorical columns.
    4. Resamples real training rows that have the selected category active.

    Args:
        data: Full training tensor of shape ``(n_rows, n_cols)``.
        categorical_groups: List of index groups corresponding to OHE categorical
            columns in the transformed space (one group per original categorical column).
    """

    def __init__(self, data: torch.Tensor, categorical_groups: list[list[int]]) -> None:
        self.data = data
        self.groups = categorical_groups
        self.n_cat_cols = len(categorical_groups)

        # Empirical category probability for each group
        self.probs: list[np.ndarray] = []
        # Pre-built per-category row index lists for fast conditioned sampling
        self._row_indices: list[list[np.ndarray]] = []

        for group in categorical_groups:
            vals = data[:, group].numpy()  # (n_rows, n_categories)
            counts = vals.sum(axis=0)
            total = counts.sum()
            self.probs.append(counts / total if total > 0 else np.ones(len(group)) / len(group))

            # For each category, store the row indices where it is active
            cat_rows = []
            for k in range(len(group)):
                active = np.where(vals[:, k] > 0.5)[0]
                cat_rows.append(active)
            self._row_indices.append(cat_rows)

        # Offsets into the flat condition vector for each categorical column
        self._offsets: list[int] = []
        offset = 0
        for group in categorical_groups:
            self._offsets.append(offset)
            offset += len(group)
        self.cond_dim: int = offset

    def sample_condvec(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Sample a batch of condition vectors.

        Returns:
            cond: ``(batch_size, cond_dim)`` — one-hot in the selected column's block.
            mask: ``(batch_size, n_cat_cols)`` — 1 for the selected column, 0 elsewhere.
            col_idxs: ``(batch_size,)`` — index of the selected categorical column.
            cat_idxs: ``(batch_size,)`` — index of the selected category within that column.
        """
        if self.n_cat_cols == 0:
            empty = torch.zeros(batch_size, 0)
            return empty, empty, np.zeros(batch_size, dtype=int), np.zeros(batch_size, dtype=int)

        # Uniformly select one categorical column per sample
        col_idxs = np.random.randint(self.n_cat_cols, size=batch_size)

        # Sample a category proportional to empirical frequency
        cat_idxs = np.array([np.random.choice(len(self.groups[c]), p=self.probs[c]) for c in col_idxs])

        cond = np.zeros((batch_size, self.cond_dim), dtype=np.float32)
        mask = np.zeros((batch_size, self.n_cat_cols), dtype=np.float32)

        for i, (c, k) in enumerate(zip(col_idxs, cat_idxs)):
            cond[i, self._offsets[c] + k] = 1.0
            mask[i, c] = 1.0

        return torch.FloatTensor(cond), torch.FloatTensor(mask), col_idxs, cat_idxs

    def sample_data_conditioned(self, batch_size: int, col_idxs: np.ndarray, cat_idxs: np.ndarray) -> torch.Tensor:
        """
        Sample real training rows conditioned on each ``(col, category)`` pair.

        For each sample, picks a random row from the training data where the
        specified category is active in the specified column. Falls back to a
        random row if no such rows exist.

        Returns:
            ``(batch_size, n_cols)`` tensor of real rows.
        """
        selected = []
        for c, k in zip(col_idxs, cat_idxs):
            active = self._row_indices[c][k]
            if len(active) == 0:
                idx = np.random.randint(len(self.data))
            else:
                idx = active[np.random.randint(len(active))].item()
            selected.append(self.data[idx])
        return torch.stack(selected)


def extract_categorical_groups(multi_column_indices: list[list[int]], columns) -> list[list[int]]:
    """
    Identify which groups in ``multi_column_indices`` are OHE categorical columns
    (as opposed to GMM component columns, whose names end in ``_c<digit>``).

    Args:
        multi_column_indices: From ``metatransformer.multi_column_indices``.
        columns: Column names (``pd.Index`` or list) of the transformed data.

    Returns:
        Subset of ``multi_column_indices`` containing only categorical groups.
    """
    categorical_groups = []
    for group in multi_column_indices:
        names = [str(columns[i]) for i in group]
        if not any(re.search(r"_c\d+$", n) for n in names):
            categorical_groups.append(group)
    return categorical_groups
