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

    # Check if we have enough data and both classes after dropna
    if len(data) < 10:
        # Not enough data to train/test
        return pd.DataFrame({f"rf_{target}_prob": []}), {"rocauc_rf": float("nan")}

    y = data[target]
    n_classes = y.nunique()
    if n_classes < 2:
        # Only one class present - ROC AUC is undefined
        return pd.DataFrame({f"rf_{target}_prob": []}), {"rocauc_rf": float("nan")}

    X = data.drop(["dob", "x3", target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42
    )

    # Check if both classes are in training set
    if y_train.nunique() < 2:
        return pd.DataFrame({f"rf_{target}_prob": []}), {"rocauc_rf": float("nan")}

    # Fit a random forest classifier on the dataset
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Get the predicted probabilities and predictions
    # Handle case where classifier only learned one class
    if rf.n_classes_ < 2:
        return pd.DataFrame({f"rf_{target}_prob": []}), {"rocauc_rf": float("nan")}

    probs = pd.DataFrame(rf.predict_proba(X_test)[:, 1], columns=[f"rf_{target}_prob"])

    rocauc = roc_auc_score(y_test, probs)

    return probs, {"rocauc_rf": rocauc}


task = Task("Random Forest on 'event'", run, supports_fairness=True, target="event")
