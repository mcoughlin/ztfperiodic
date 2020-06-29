
import numpy as np

from xgboost import XGBClassifier
import xgboost as xgb

def classify(algorithm, features, modelFile=None):

    if algorithm == "xgboost":
        dtest = xgb.DMatrix(features)
        ################ Actual inferencing

        loaded_model = xgb.Booster()

        loaded_model.load_model(modelFile)
        # And use it for predictions.
        pred = loaded_model.predict(dtest)

    return pred
