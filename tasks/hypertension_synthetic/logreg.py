import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nhssynth.modules.evaluation.tasks import Task


def run(dataset) -> tuple[pd.DataFrame, dict]:
    """Logistic regression classifier for predicting Hypertension_Diagnosis."""
    target = "Hypertension_Diagnosis"

    data = dataset.dropna()

    # Check if we have enough data
    if len(data) < 10:
        return pd.DataFrame({f"lr_{target}_prob": []}), {"rocauc_lr": float("nan")}

    y = data[target]
    n_classes = y.nunique()
    if n_classes < 2:
        # Only one class present - ROC AUC is undefined
        return pd.DataFrame({f"lr_{target}_prob": []}), {"rocauc_lr": float("nan")}

    # Drop non-feature columns (datetime columns and target)
    drop_cols = ["Date_of_Birth", "Medication_Start_Date", "Last_Followup_Date", target]
    X = data.drop([c for c in drop_cols if c in data.columns], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42
    )

    # Check if both classes are in training set
    if y_train.nunique() < 2:
        return pd.DataFrame({f"lr_{target}_prob": []}), {"rocauc_lr": float("nan")}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Get the predicted probabilities
    probs = pd.DataFrame(lr.predict_proba(X_test)[:, 1], columns=[f"lr_{target}_prob"])

    rocauc = roc_auc_score(y_test, probs)

    return probs, {"rocauc_lr": rocauc}


task = Task("Logistic Regression on Hypertension", run, supports_fairness=True, target="Hypertension_Diagnosis")
