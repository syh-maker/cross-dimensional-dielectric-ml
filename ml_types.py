from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class DropCollinearFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        if X.shape[1] > 1:
            correlation = X.corr().abs()
            upper_triangle = correlation.where(
                np.triu(np.ones(correlation.shape), k=1).astype(bool)
            )
            self.to_drop_ = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.threshold)
            ]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")


class RobustPreprocessor(BaseEstimator, TransformerMixin):
    """Compatibility transformer stored in the published model artifacts."""

    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        self.imputer = SimpleImputer(strategy="median")
        imputed = self.imputer.fit_transform(X)
        self.scaler = RobustScaler()
        self.scaler.fit(imputed)
        return self

    def transform(self, X):
        aligned = X.reindex(columns=self.cols_, fill_value=np.nan)
        imputed = self.imputer.transform(aligned)
        transformed = self.scaler.transform(imputed)
        return type(X)(transformed, columns=self.cols_, index=X.index)


@dataclass
class PreprocessBundle:
    input_cols: List[str]
    non_empty_cols: List[str]
    imputer: SimpleImputer
    variance: VarianceThreshold
    variance_cols: List[str]
    scaler: RobustScaler
    final_cols_before_drop: List[str]
    dropper: DropCollinearFeatures
    output_cols: List[str]
