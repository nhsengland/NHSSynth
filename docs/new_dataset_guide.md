# Applying the Pipeline to a New Dataset

This guide outlines the steps to run the NHSSynth pipeline on a new dataset.

## Prerequisites

- A CSV file containing your dataset
- Understanding of your data's column types and constraints

## Step 1: Create the Metadata File

Create a YAML file (e.g., `data/your_dataset_metadata.yaml`) defining your columns and constraints.

### Column Definitions

```yaml
columns:
  # Datetime column
  date_column:
    dtype:
      name: datetime64
      floor: S
      format: '%Y-%m-%d %H:%M:%S'
    categorical: false

  # Categorical column (encoded as integers)
  category_column:
    categorical: true
    dtype: int64

  # Numerical column (float)
  numeric_column:
    dtype: float64
    rounding_scheme: 0.00001  # Optional: decimal precision

  # Numerical column (integer)
  integer_column:
    dtype: int64
```

### Constraints

```yaml
constraints:
  # Range constraints
  - "column_name in (min, max)"

  # Comparison constraints
  - "column_a > column_b"

  # Binary field validation
  - "binary_column in (0,1)"
```

## Step 2: Create the Pipeline Configuration

Create a YAML config file (e.g., `config/your_pipeline.yaml`):

```yaml
seed: 1
experiment_name: your_experiment
run_type: pipeline

dataloader:
  collapse_yaml: true
  write_csv: true

model:
  architecture:
    - DPVAE
  num_epochs:
    - 50
  patience: 10
  target_epsilon:
    - 1.0
  max_grad_norm: 5.0
  secure_mode: false
  repeats: 1

evaluation:
  # Downstream tasks (requires task files in tasks/your_dataset/)
  downstream_tasks: true  # Set false if no tasks defined
  fairness: true          # Requires downstream_tasks: true
  protected_attributes:
    - demographic_column_1
    - demographic_column_2

  # Metrics (select as needed)
  column_shape_metrics:
    - KSComplement
    - TVComplement
  column_similarity_metrics:
    - StatisticSimilarity
    - CorrelationSimilarity
  boundary_metrics:
    - BoundaryAdherence
  coverage_metrics:
    - RangeCoverage
    - CategoryCoverage

  # Privacy metrics
  key_categorical_fields:
    - quasi_identifier_1
  sensitive_categorical_fields:
    - sensitive_outcome
  key_numerical_fields:
    - quasi_identifier_numeric
  sensitive_numerical_fields:
    - sensitive_numeric_value
```

## Step 3: Create Downstream Tasks (Optional)

To enable fairness metrics, create a task file at `tasks/your_dataset/task_name.py`:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nhssynth.modules.evaluation.tasks import Task


def run(dataset) -> tuple[pd.DataFrame, dict]:
    target = "your_target_column"
    data = dataset.dropna()

    if len(data) < 10 or data[target].nunique() < 2:
        return pd.DataFrame({f"lr_{target}_prob": []}), {"rocauc_lr": float("nan")}

    # Drop non-feature columns (datetime, target, etc.)
    drop_cols = ["datetime_col", target]
    X = data.drop([c for c in drop_cols if c in data.columns], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    probs = pd.DataFrame(lr.predict_proba(X_test)[:, 1], columns=[f"lr_{target}_prob"])
    rocauc = roc_auc_score(y_test, probs)

    return probs, {"rocauc_lr": rocauc}


task = Task("Task Name", run, supports_fairness=True, target="your_target_column")
```

## Step 4: Run the Pipeline

```bash
nhssynth pipeline \
  -c config/your_pipeline.yaml \
  -d data/your_dataset.csv \
  -m data/your_dataset_metadata.yaml
```

## Step 5: View Results

Launch the dashboard to explore evaluation results:

```bash
nhssynth dashboard -e experiments/your_experiment/
```

## Checklist

- [ ] Metadata file created with all columns defined
- [ ] Data types correctly specified (datetime64, int64, float64)
- [ ] Categorical columns marked with `categorical: true`
- [ ] Constraints added for range validation and logical relationships
- [ ] Pipeline config created with appropriate metrics
- [ ] Privacy fields selected (key = quasi-identifiers, sensitive = protected values)
- [ ] (Optional) Downstream task created for fairness evaluation
- [ ] Protected attributes selected for fairness analysis
