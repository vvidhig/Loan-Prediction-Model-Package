from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import numpy as np

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        # Compute the mean for each variable
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        # Create a copy of the dataframe
        X = X.copy()
        # Fill NaNs with the computed means
        for col in self.variables:
            X[col] = X[col].fillna(self.mean_dict[col])  # Corrected to avoid inplace assignment
        return X


class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mode_dict = {}

    def fit(self, X, y=None):
        # Compute the mode for each variable
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        # Create a copy of the dataframe
        X = X.copy()
        # Fill NaNs with the computed modes
        for col in self.variables:
            X[col] = X[col].fillna(self.mode_dict[col])  # Corrected to avoid inplace assignment
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns = self.variables)
        return X

class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_modify = None, variables_to_add = None):
        self.variables_to_modify = variables_to_modify
        self.variables_to_add = variables_to_add
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for feature in self.variables_to_modify:
            X[feature] = X[feature] + X[self.variables_to_add]
        return X

class LabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    def fit(self, X,y):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X

class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables  
    def fit(self,X,y=None):
        return self 
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X