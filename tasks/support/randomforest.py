import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nhssynth.modules.evaluation.tasks import Task


def run(dataset) -> tuple[pd.DataFrame, dict]:
    # Split the dataset into features and target
    target = "event"

    data = dataset.dropna()

    X, y = data.drop(["dob", "x3", target], axis=1), data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42
    )

    # Fit a random forest classifier on the dataset
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Get the predicted probabilities and predictions
    probs = pd.DataFrame(rf.predict_proba(X_test)[:, 1], columns=[f"rf_{target}_prob"])

    rocauc = roc_auc_score(y_test, probs)

    return probs, {"rocauc_rf": rocauc}


task = Task("Random Forest on 'event'", run, supports_aequitas=True)
