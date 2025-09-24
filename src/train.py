import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

from src.data import load_dataframe, get_xy, split_train_valid
from src.models import svm_rbf, logreg, rf

def _save_rf_importance(model, feature_names, out_dir, model_name):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fi = pd.DataFrame({
        "feature": np.array(feature_names)[order],
        "importance": importances[order]
    })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"feature_importance_{model_name}.csv")
    fi.to_csv(csv_path, index=False)

    # Plot (horizontal bar)
    plt.figure(figsize=(7, 5))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"feature_importance_{model_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return csv_path, fig_path

def train_and_eval(data_path, model_name="svm", smote=False, out_dir="reports"):
    df = load_dataframe(data_path)
    X, y = get_xy(df)  # uses only your 11 features
    feature_names = X.columns.tolist()

    X_train, X_valid, y_train, y_valid = split_train_valid(X, y, test_size=0.2)

    if smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    if model_name == "svm":
        model = svm_rbf(C=10.0, gamma="scale", class_weight=None)
    elif model_name == "logreg":
        model = logreg(max_iter=1000, class_weight=None)
    elif model_name == "rf":
        model = rf(n_estimators=300, max_depth=None, class_weight=None, random_state=42)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    acc = float(accuracy_score(y_valid, preds))
    f1w = float(f1_score(y_valid, preds, average="weighted"))
    cm = confusion_matrix(y_valid, preds)

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, f"cm_{model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Metrics
    metrics = {
    "model": str(model_name),
    "accuracy": float(acc),
    "f1_weighted": float(f1w),
    "labels": [int(l) for l in np.unique(y_valid)],  
    "confusion_matrix": cm.astype(int).tolist()      
    }

    metrics_path = os.path.join(out_dir, f"metrics_{model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    report = classification_report(y_valid, preds, digits=3)
    report_path = os.path.join(out_dir, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Feature importance for RF
    fi_csv = None
    fi_png = None
    if model_name == "rf":
        fi_csv, fi_png = _save_rf_importance(model, feature_names, out_dir, model_name)

    return {
        "metrics_path": metrics_path,
        "confusion_matrix_png": cm_path,
        "report_path": report_path,
        "feature_importance_csv": fi_csv,
        "feature_importance_png": fi_png
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cleaned_wine_dataset.csv", help="CSV path")
    ap.add_argument("--model", default="svm", choices=["svm","logreg","rf"])
    ap.add_argument("--smote", action="store_true", help="Apply SMOTE to train split")
    ap.add_argument("--out_dir", default="reports", help="Output directory for metrics and figures")
    args = ap.parse_args()

    out = train_and_eval(args.data, args.model, args.smote, args.out_dir)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
