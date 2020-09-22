
def classify(algorithm, features, modelFile=None):

    if algorithm == "xgboost":
        import xgboost as xgb

        dtest = xgb.DMatrix(features)
        ################ Actual inferencing

        loaded_model = xgb.Booster()

        loaded_model.load_model(modelFile)
        # And use it for predictions.
        pred = loaded_model.predict(dtest)
    elif algorithm == "dnn":
        import json
        import numpy as np
        from copy import deepcopy
        from tensorflow.keras.models import load_model

        model = load_model(modelFile)
        normFile = "/".join(modelFile.split("/")[:-1]) + "/norms.20200615.json"
        with open(normFile, 'r') as f:
            norms = json.load(f)

        dff = deepcopy(features)
        dmdt = np.expand_dims(np.array([d for d in dff['dmdt'].values]),
                              axis=-1)
        dff.drop(columns='dmdt', inplace=True)

        # apply norms
        for feature, norm in norms.items():
            if not feature in dff.columns: continue
            dff[feature] /= norm
        dff.fillna(0, inplace=True)

        pred = model.predict([dff.values, dmdt], verbose=False).flatten()

    return pred
