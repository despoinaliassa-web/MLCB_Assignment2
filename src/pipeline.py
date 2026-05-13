import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import (matthews_corrcoef, f1_score, roc_auc_score, 
                             balanced_accuracy_score, recall_score, 
                             precision_score, confusion_matrix, average_precision_score)

class RepeatedNestedCV:
    def __init__(self, dataset, estimators, param_spaces, R=10, N=5, K=3, seed=42):
        """
        Implementation of the Repeated Nested Cross-Validation pipeline
        Separation of training and test data (No Leakage)
        """
        self.X = dataset.drop(columns=['target'])
        self.y = dataset['target']
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.R = R
        self.N = N
        self.K = K
        self.seed = seed
        self.results = []

        # Feature Grouping based on EDA
        # We explicitly exclude 'chol' and 'fbs' from these lists to perform Feature Selection
        self.num_features = ['age', 'trestbps', 'thalach', 'oldpeak']
        self.cat_features = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']

    def _create_pipeline(self, estimator):
        """
        Creates a ColumnTransformer pipeline. 
        Numerical: Median Imputation + Robust Scaling (for outliers).
        Categorical: Mode Imputation + OneHot Encoding.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Feature Selection: 'remainder=drop' removes any columns 
        # not mentioned in num_features or cat_features (like 'chol' and 'fbs')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ],
            remainder='drop'
        )

        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])

    def _get_bootstrap_ci(self, data, n_iterations=1000, ci=95):
        """
        Helper method to compute 95% Confidence Intervals for the median (Task 3.3)
        """
        stats = []
        for i in range(n_iterations):
            # Bootstrap sample with replacement
            sample = resample(data, replace=True, random_state=self.seed + i)
            stats.append(np.median(sample))
        lower = np.percentile(stats, (100 - ci) / 2)
        upper = np.percentile(stats, 100 - (100 - ci) / 2)
        return lower, upper

    def run(self, tune_hyperparameters=True, inner_score='f1'):
        """
        Executes rnCV 
        inner_score: Metric for hyperparameter tuning (e.g., 'f1', 'roc_auc', 'balanced_accuracy')
        """
        self.results = [] 
        for r in range(self.R):
            # Outer loop split 
            outer_cv = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=self.seed + r)
            
            for train_idx, test_idx in outer_cv.split(self.X, self.y):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
                
                for name, est in self.estimators.items():
                    pipe = self._create_pipeline(est)
                    
                    if tune_hyperparameters and self.param_spaces[name]:
                        # Inner loop for hyperparameter tuning 
                        inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.seed + r)
                        grid = GridSearchCV(pipe, self.param_spaces[name], cv=inner_cv, scoring=inner_score, n_jobs=-1)
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                        params = str(grid.best_params_)
                    else:
                        # Baseline comparison with default parameters
                        pipe.fit(X_train, y_train)
                        best_model = pipe
                        params = "default"

                    # Predictions for assessment
                    preds = best_model.predict(X_test)
                    # 1. Confusion Matrix calculation for το Specificity 
                    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                    # 2. Probability calculation for AUC and PRAUC
                    probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

                    # Save all required metrics
                    self.results.append({
                        'Repetition': r, 'Algorithm': name,
                        'MCC': matthews_corrcoef(y_test, preds),
                        'AUC': roc_auc_score(y_test, probs) if probs is not None else np.nan,
                        'BA': balanced_accuracy_score(y_test, preds),
                        'F1': f1_score(y_test, preds),
                        'Recall': recall_score(y_test, preds),
                        'Specificity': spec,
                        'Precision': precision_score(y_test, preds),
                        'PRAUC': average_precision_score(y_test, probs) if probs is not None else np.nan,
                        'Best_Params': params
                    })
        return pd.DataFrame(self.results)

    def get_statistical_summary(self, df_results):
        """
        Computes median performance and 95% Confidence Intervals for all metrics (Task 3.3).
        """
        summary = []
        metrics = ['MCC', 'AUC', 'BA', 'F1', 'Recall', 'Specificity', 'Precision', 'PRAUC']
        for alg in df_results['Algorithm'].unique():
            alg_data = df_results[df_results['Algorithm'] == alg]
            alg_sum = {'Algorithm': alg}
            for m in metrics:
                vals = alg_data[m].dropna()
                low, high = self._get_bootstrap_ci(vals)
                alg_sum[f'{m}_median'] = vals.median()
                alg_sum[f'{m}_95CI'] = (round(low, 3), round(high, 3))
            summary.append(alg_sum)
        return pd.DataFrame(summary)