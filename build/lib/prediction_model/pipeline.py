from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ("DomainProcessing", pp.DomainProcessing(variables_to_modify=config.FEATURE_TO_MODIFY,
          variables_to_add = config.FEATURE_TO_ADD)),
        ("MeanImptation", pp.MeanImputer(variables = config.NUM_FEATURES)),
        ("ModeImputations", pp.ModeImputer(variables = config.CAT_FEATURES)),
        ("DropFeatures", pp.DropColumns(variables = config.DROP_FEATURES)),
        ("LabelEncoder", pp.LabelEncoder(variables = config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)
