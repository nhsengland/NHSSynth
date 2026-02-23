# Changelog

All notable changes to NHSSynth are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- **GAN model (WGAN-GP)** — new `GAN` architecture available via `model.architecture: [GAN]` in pipeline config.
  - Generator and critic are both configurable MLPs (`noise_dim`, `generator_*`, `discriminator_*` params).
  - Critic uses raw Wasserstein scores (no sigmoid); gradient penalty enforces the Lipschitz constraint (`lambda_gradient_penalty`, default 10).
  - Supports optional conditional inputs (`n_units_conditional`).
- **DPGAN model** — differentially private variant of the GAN (`model.architecture: [DPGAN]`).
  - DP applied to the discriminator only (the sole component that processes real data), following the standard DP-GAN approach.
  - Accepts the same `target_epsilon`, `target_delta`, `max_grad_norm`, `secure_mode` arguments as `DPVAE`.
- **Shared MLP backbone** (`src/nhssynth/modules/model/common/mlp.py`) — configurable multi-layer perceptron used by the GAN generator and critic.
  - Supports batch normalisation, dropout, residual skip connections, and a choice of activation functions.
- GAN and DPGAN example config comments added to `config/test_pipeline.yaml`.

### Fixed

- **CI: Python version** — all GitHub Actions workflows and `tox.ini` standardised to Python 3.10 (was inconsistently set to 3.11 in some workflows).
- **CI: Poetry version** — Poetry pinned to 1.8.5 across all workflows to avoid Poetry 2.x incompatibilities.
- **CI: `setuptools` missing** — added `setuptools>=65.0.0` to dev dependencies to fix `No module named 'pkg_resources'` errors in CI.
- **CI: dynamic badges** — all four badge update steps set to `continue-on-error: true`; gist ID updated to the project owner's gist.
- **Dead code removed** — orphaned methods in `metatransformer.py` (`_ensure_binary_or_in`, `_get_pool_for`, `_fold_into_interval`, `_bootstrap_into_interval`) and their associated unused imports were removed, resolving several ruff `F821` errors.

---

## [0.10.2] — prior release

See git log for earlier history.
