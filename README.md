# Wine Quality Prediction (Ordinal Learning)

**Goal**: Predict wine quality (ordinal labels) from physicochemical features and benchmark multiple models: SVM (RBF), Logistic Regression, and Random Forest. Additional experiments with Ordinal SVM and Ordinal Logistic Regression are available in the notebook and detailed report.

**Why it matters**: Wine quality scores are *ordered* (ordinal), not just categorical. Treating them with ordinal-aware methods or evaluating adjacency in misclassifications better reflects real-world behavior.

---

## Repository Contents
```
wine-quality-ordinal/
├─ README.md                  # You are here
├─ requirements.txt           # Dependencies
├─ data/cleaned_wine_dataset.csv
├─ notebooks/01_eda_and_modeling.ipynb   # Detailed workflow (EDA + modeling)
├─ src/                       # Training scripts (CLI)
│   ├─ train.py
│   ├─ model.py
│   └─ data.py
├─ reports/                   # Outputs (metrics, figures, reports)
│   ├─ metrics_*.json
│   ├─ classification_report_*.txt
│   ├─ figures/cm_*.png
│   ├─ feature_importance_rf.csv
│   └─ feature_importance_rf.png
├─ Wine_Quality_Ordinal.pdf   # Full project report (kept visible at repo root)
└─ .gitignore
```

---

## Setup & Installation
```bash
# Create and activate virtual environment (example: venv)
python -m venv wine_env
wine_env\Scripts\activate   # On Windows
source wine_env/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart
### Option 1: Explore via Notebook (detailed analysis)
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```
The notebook includes:
- Exploratory Data Analysis (EDA)
- Feature importance via Random Forest
- Experiments with Ordinal SVM and Ordinal Logistic Regression (from mord)
- Future scope discussions (with additional notes and ideas not covered in the main report)

### Option 2: Run via Command Line (reproducible training)
```bash
# Train & evaluate SVM (RBF), save metrics + confusion matrix
python -m src.train --data data/cleaned_wine_dataset.csv --model svm --out_dir reports

# Train Logistic Regression
python -m src.train --data data/cleaned_wine_dataset.csv --model logreg --out_dir reports

# Train Random Forest
python -m src.train --data data/cleaned_wine_dataset.csv --model rf --out_dir reports
```

Each run saves:
- Metrics JSON → `reports/metrics_<model>.json`
- Confusion matrix PNG → `reports/figures/cm_<model>.png`
- Classification report TXT → `reports/classification_report_<model>.txt`
- Feature importance (RF only) → `reports/feature_importance_rf.csv/png`

---

## Results (TL;DR)
### Random Forest (best performing)
- **Accuracy**: 0.693
- **F1-weighted**: 0.681
- Confusion matrix → `reports/figures/cm_rf.png`
- Feature importance table available → `reports/feature_importance_rf.csv`

### SVM (RBF)
- **Accuracy**: 0.591
- **F1-weighted**: 0.575

### Logistic Regression
- **Accuracy**: 0.535
- **F1-weighted**: 0.506

### Feature Importance (Random Forest)
| Feature              | Importance |
|----------------------|-----------:|
| alcohol              | 0.123 |
| density              | 0.110 |
| volatile acidity     | 0.089 |
| total sulfur dioxide | 0.077 |
| chlorides            | 0.064 |
| free sulfur dioxide  | 0.058 |
| sulphates            | 0.052 |
| pH                   | 0.046 |
| residual sugar       | 0.039 |
| citric acid          | 0.021 |
| fixed acidity        | 0.015 |

---

## Future Work
- Implement **Ordinal Logistic Regression** and **Ordinal SVM** into the CLI (`train.py`) for consistent comparison.
- Explore **SVR + discretization** as an ordinal approach.
- Test alternate imbalance handling strategies beyond SMOTE.
- Expand hyperparameter search.
- Add ordinal-specific evaluation metrics (e.g., Kendall’s Tau, Quadratic Weighted Kappa).
- Refer to the notebook for **additional future work ideas** that go beyond the main report.

---

## Notes for Recruiters
- **Reproducibility**: Clear structure, reproducible training commands, and outputs.
- **Depth**: Full **report (Wine_Quality_Ordinal.pdf)** and **detailed notebook** available for in-depth review, with extra future work included.
- **Breadth**: Covers EDA, feature engineering, model benchmarking, and reflections on ordinal vs multiclass approaches.
- **Portfolio-ready**: Quick to skim at the repo root, while offering detailed artifacts for deeper inspection.

---

## Team Members
- **Manisha Kandal** – Lead Developer / Core Logic & Architecture  
- **Pratyush Nakka** – Developer / Integration & Maintenance  
