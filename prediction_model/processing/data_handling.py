import os
import pandas as pd
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config

#Loading the dataset
def load_dataset(filename):
    filepath = os.path.join(config.DATAPATH, filename)
    _data = pd.read_csv(filepath)
    return _data

#Serialization
def save_pipeline(pipeline):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline, save_path)
    print("Model has been saved under the name ", config.MODEL_NAME)
    
#Deserialization
def load_pipeline(pipeline):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("Model has been loaded successfully")
    return model_loaded 