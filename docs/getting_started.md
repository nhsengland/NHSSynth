# Getting Started

## Running an experiment

This package offers two easy ways to run reproducible and highly-configurable experiments. The following sections describe how to use each of these two methods.

### Via the CLI

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

This will run a full pipeline experiment on the `support` dataset in the `data` directory. The outputs of the experiment will be recorded in a folder named `test` (corresponding to the experiment name) in the `experiments` directory.

In total, three different model architectures will be trained three times each with their default configurations. The resulting generated synthetic datasets will be evaluated via the downstream tasks in `tasks/support` alongside the metrics specified in the command. A dashboard will then be built automatically to exhibit the results.

The components of the run are persistent to the experiment's folder. Suppose you have already run this experiment and want to add some new evaluations. You do not have to re-run the entire experiment, you can simply run:

```bash
nhssynth evaluation -e test -d support -s 123 --coverage-metrics RangeCoverage CategoryCoverage
nhssynth dashboard -e test -d support
```

This will regenerate the dashboard with a different set of metrics corresponding to the arguments passed to `evaluation`. Note that the `--experiment-name` and `--dataset` arguments are required for all commands, as they are used to identify the experiment and ensure reproducibility.

### Via a configuration file

A `yaml` configuration file placed in the `config` folder can be used to get the same result as the above:

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

Once saved as `run_pipeline.yaml` in the `config` directory, the package can be run under the configuration laid out in the file via:

```bash
nhssynth config -c run_pipeline
```

Note that if you run via the [CLI](#via-cli), you can add the `--save-config` flag to your command to save the configuration file in the `experiments/test` (or whatever the `--experiment-name` has been set to) directory. This allows for easy reproduction of an experiment at a later date or on someone else's computer through sharing the configuration file with them.

## Setting up a dataset's metadata

For each dataset you wish to work with, it is advisable to setup a corresponding metadata file. The package will infer this when information is missing (and you can then tweak it). The reason we suggest specifying metadata in this way is because Pandas / Python are in general bad at interpreting CSV files, particularly the specifics of datatypes, date objects and so on.

To do this, we must create a metadata `yaml` file in the dataset's directory. For example, for the `support` dataset, this file is located at `data/support_metadata.yaml`. By default, the package will look for a file with the same name as the dataset in the dataset's directory, but with `_metadata` appended to the end. *This is configurable like most other filenaming conventions via the CLI.*

The metadata file is split into two sections: `columns` and `constraints`. The former specifies the nature of each column in the dataset, whilst the latter specifies any constraints that should be enforced on the dataset.

### Column metadata

Again, we refer to the `support` dataset's metadata file as an example:

```yaml
columns:
  dob:
    dtype:
      name: datetime64
      floor: S
  x1:
    categorical: true
    dtype: int64
  x2:
    categorical: true
    dtype: int64
  x3:
    categorical: true
  x4:
    categorical: true
    dtype: int64
  x5:
    categorical: true
    dtype: int64
  x6:
    categorical: true
    dtype: int64
  x7:
    dtype: int64
  x8:
    dtype: float64
    missingness:
      impute: mean
  x9:
    dtype: int64
  x10:
    dtype:
      name: float64
      rounding_scheme: 0.1
  x11:
    dtype: int64
  x12:
    dtype: float64
  x13:
    dtype: float64
  x14:
    dtype: float64
  duration:
    dtype: int64
  event:
    categorical: true
    dtype: int64
```

For each column in the dataset, we specify the following:

- It's `dtype`, this can be any `numpy` data type or a datetime type.
  - In the case of a datetime type, we also specify the `floor` (i.e. the smallest unit of time that we care about). In general this should be set to match the smallest unit of time in the dataset.
  - In the case of a `float` type, we can also specify a `rounding_scheme` to round the values to a certain number of decimal places, again this should be set according to the rounding applied to the column in the real data, or if you want to round the values for some other reason.
- Whether it is `categorical` or not. If a column is not categorical, you don't need to specify this. A column is inferred as `categorical` if it has less than 10 unique values or is a string type.
- If the column has missing values, we can specify how to deal with them by specifying a `missingness` strategy. In the case of the `x8` column, we `impute` the missing values with the column's `mean`. If you don't specify this, the CLI or configuration file's specified global missingness strategy will be applied instead (this defaults to the augment strategy which model's the missingness as a separate level in the case of categorical features, or as a separate cluster in the case of continuous features).

### Constraints

The second part of the metadata file specifies any constraints that should be enforced on the dataset. These can be a relative constraint between two columns, or a fixed one via a constant on a single column. For example, the `support` dataset's constraints are as follows (note that these are arbitrarily defined and do not necessarily reflect the real data):

```yaml
constraints:
  - "x10 in (0,100)"
  - "x12 in (0,100)"
  - "x13 in (0,100)"
  - "x10 <= x12"
  - "x12 < x13"
  - "x10 < x13"
  - "x8 > x10"
  - "x8 > x12"
  - "x8 > x13"
  - "x11 > 100"
  - "x12 > 10"
```

The function of these constraints is fairly self-explanatory: The package ensures the constraints are feasible and minimises them before applying transformations to ensure that they will be satisfied in the synthetic data as well. When a column does not meet a feasible constraint in the real data, we assume that this is intentional and use the violation as a feature upon which to generate synthetic data that also violates the constraint.

There is a further constraint `fixcombo` that only applies to categorical columns. This suggests that only existing combinations of two or more categorical columns should be generated, i.e. the columns can be collapsed into a single composite feature. I.e. if we have a column for pregnancy, and another for sex, we may only want to allow three categories, 'male:not-pregnant', 'female:pregnant', 'female:not-pregnant'. This is specified as follows:

```yaml
constraints:
  - "pregnancy fixcombo sex"
```

In conclusion then, we support the following constraint types:

- `fixcombo` for categorical columns
- `<` and `<` for non-categorical columns
- `>=` and `<=` for non-categorical columns
- `in` for non-categorical columns, which is effectively two of the above constraints combined. I.e. `x in [a, b)` is equivalent to `x >= a and x < b`.

Once this metadata is setup, you are ready to run your experiment.

## Evaluation

Once models have been trained and synthetic datasets generated, we leverage evaluations from [SDMetrics](https://docs.sdv.dev/sdmetrics), [Aequitas](http://aequitas.dssg.io), the NHS' internal [SynAdvSuite](https://github.com/nhsengland/SynAdvSuite) (at current time you must request access to this repository to use the privacy-related attacks it implements), and also offer a facility for the [custom specification of downstream tasks](downstream_tasks.md). These evaluations are then aggregated into a dashboard for ease of comparison and analysis.

See the relevant documentation for each of these packages for more information on the metrics they offer.
