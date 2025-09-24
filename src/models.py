from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mord import LogisticIT
import numpy as np

def olr(max_iter=1000):

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticIT(max_iter=max_iter))
    ])

class OrdinalSVMRBF(BaseEstimator, ClassifierMixin):

    def __init__(self, C=100.0, gamma=0.1, class_weight=None, random_state=None):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        self.classes_ = None
        self.clfs_ = None

    def fit(self, X, y):
        
        self.classes_ = np.array(sorted(np.unique(y)))
        K = len(self.classes_)
        self.clfs_ = []
        # For each cut k (all but the max class), train y > k vs y <= k
        for k_idx in range(K - 1):
            cut = self.classes_[k_idx]
            y_bin = (y > cut).astype(int)
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf",
                            C=self.C,
                            gamma=self.gamma,
                            class_weight=self.class_weight,
                            probability=False,
                            random_state=self.random_state))
            ])
            clf.fit(X, y_bin)
            self.clfs_.append(clf)
        return self

    def predict(self, X):

        votes = np.zeros((X.shape[0],), dtype=int)
        for clf in self.clfs_:
            sv = clf.named_steps["svm"]
            try:
                dec = clf.decision_function(X)  
                votes += (dec > 0).astype(int)
            except Exception:
                votes += clf.predict(X).astype(int)

        return self.classes_[votes]
        
def ordinal_svm_rbf(C=100.0, gamma=0.1, class_weight=None):
    return OrdinalSVMRBF(C=C, gamma=gamma, class_weight=class_weight)


def svm_rbf(C=10.0, gamma="scale", class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight))
    ])

def logreg(max_iter=1000, class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight, multi_class="auto"))
    ])

def rf(n_estimators=300, max_depth=None, class_weight=None, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight=class_weight, random_state=random_state
    )
