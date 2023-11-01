# Running an experiment

This package offers two easy ways to run reproducible and highly-configurable experiments. The following sections describe how to use each of these two methods.

## Via the CLI

The CLI is the easiest way to quickly run an experiment. It is designed to be as simple as possible, whilst still offering a high degree of configurability. An example command to run a full pipeline experiment is:

```bash
nhssynth pipeline \
    --experiment-name test \
    --dataset support \
    --seed 123 \
    --architecture DPVAE PATEGAN DECAF \
    --repeats 3 \
    --downstream-tasks \
    --column-similarity-metrics CorrelationSimilarity ContingencySimilarity \
    --column-shape-metrics KSComplement TVComplement \
    --boundary-metrics BoundaryAdherence \
    --synthesis-metrics NewRowSynthesis \
    --divergence-metrics ContinuousKLDivergence DiscreteKLDivergence
```

This will run a full pipeline experiment on the `support` dataset in the `datasets` directory. The outputs of the experiment will be recorded in a folder named `test` (corresponding to the experiment name) in the `experiments` directory.

In total, three different model architectures will be trained three times each with their default configurations. The resulting generated synthetic datasets will be evaluated via the downstream tasks in `tasks/support` alongside the metrics specified in the command. A dashboard will then automatically be generated exhibiting these results.

Suppose you have already run this experiment and want to add some new evaluations. You do not have to re-run the entire experiment, you can simply run:

```bash
nhssynth evaluation -e test -d support -s 123 --coverage-metrics RangeCoverage CategoryCoverage
nhssynth dashboard -e test -d support
```

This will regenerate the dashboard with a different set of metrics corresponding to the arguments passed to `evaluation`. Note that the `--experiment-name` and `--dataset` arguments are required for all commands, as they are used to identify the experiment and ensure reproducibility.

## Via a configuration file

A `yaml` configuration file can be used to get the same result as the above:

```yaml
seed: 123
experiment_name: test
run_type: pipeline
model:
  architecture:
    - DPVAE
    - DPGAN
    - DECAF
  max_grad_norm: 5.0
  secure_mode: false
  repeats: 4
evaluation:
  downstream_tasks: true
  column_shape_metrics:
  - KSComplement
  - TVComplement
  column_similarity_metrics:
  - CorrelationSimilarity
  - ContingencySimilarity
  boundary_metrics:
  - BoundaryAdherence
  synthesis_metrics:
  - NewRowSynthesis
  divergence_metrics:
  - ContinuousKLDivergence
  - DiscreteKLDivergence
```

Once saved as `run_pipeline.yaml` in the `config` directory, this configuration file can be run via:

```bash
nhssynth config -c run_pipeline
```

Note that if you run via the [CLI](#via-cli), you can add the `--save-config` flag to your command to save the configuration file in the `experiments/test` (or whatever the experiment name is) directory. That way you can reproduce the experiment easily and share its full configuration with others.
