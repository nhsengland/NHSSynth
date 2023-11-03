# Defining a downstream task

It is likely that a synthetic dataset may be associated with specific modelling efforts or metrics that are not included in the general suite of evaluation tools supported more explicitly by this package. Additionally, analyses on model outputs for bias and fairness provided via [Aequitas](http://aequitas.dssg.io) require some basis of predictions on which to perform the analysis. For these reasons, we provide a simple interface for defining a custom downstream task.

All downstream tasks are to be located in a folder named `tasks` in the working directory of the project, with subfolders for each dataset, i.e. the tasks associated with the `support` dataset should be located in the `tasks/support` directory.

The interface is then quite simple:

- There should be a function called `run` that takes a single argument: `dataset` (additional arguments could be provided with some further configuration if there is a need for this)
- The `run` function should fit a model and / or calculate some metric(s) on the dataset.
- It should then return predicted probabilities for the outcome variable(s) in the dataset and a dictionary of metrics.
- The file should contain a top-level variable containing an instantiation of the `nhssynth` `Task` class.

See the example below of a logistic regression model fit on the `support` dataset with the `event` variable as the outcome and `rocauc` as the metric of interest:

```python hl_lines="7 10 28 31"
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nhssynth.modules.evaluation.tasks import Task


def run(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Split the dataset into features and target
    target = "event"

    data = dataset.dropna()
    X, y = data.drop(["dob", "x3", target], axis=1), data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42
    )

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Get the predicted probabilities and predictions
    probs = pd.DataFrame(lr.predict_proba(X_test)[:, 1], columns=[f"lr_{target}_prob"])

    rocauc = roc_auc_score(y_test, probs)

    return probs, {"rocauc_lr": rocauc}


task = Task("Logistic Regression on 'event'", run, supports_aequitas=True)
```

Note the highlighted lines above:

1. The `Task` class has been imported from `nhssynth.modules.evaluations.tasks`
2. The `run` function should accept one argument and return a tuple
3. The second element of this tuple should be a dictionary labelling each metric of interest (this name will be used in the dashboard as identification so ensure it is unique to the experiment)
4. The `task` should be instantiated with a name, the `run` function and a boolean indicating whether the task supports Aequitas analysis, if the task does *not* support Aequitas analysis, then the first element of the tuple will not be used and `None` can be returned instead.

The rest of this file can contain any arbitrary code that runs within these constraints, this could be a simple model as above, or a more complex pipeline of transformations and models to match a pre-existing workflow.
