
def classify(algorithm, features, modelFile=None, normFile=None):

    if algorithm == "xgboost":
        import xgboost as xgb
        ################ Actual inferencing
        loaded_model = xgb.Booster()
        loaded_model.load_model(modelFile)
        # And use it for predictions.
        pred = loaded_model.predict(features)

    elif algorithm == "dnn":
        import json
        import numpy as np
        from copy import deepcopy
        from tensorflow.keras.models import load_model

        model = load_model(modelFile)
        with open(normFile, 'r') as f:
            norms = json.load(f)

        dff = deepcopy(features)
        #dmdt = np.expand_dims(np.array([d for d in dff['dmdt'].values]),
        #                      axis=-1)
        dmdt = []
        for i in dff.itertuples():
            var = np.asarray(dff['dmdt'][i.Index])
            if not var.shape == (26,26):
                var = np.zeros((26,26))
            dmdt.append(var)
        dmdt = np.dstack(dmdt)
        dmdt = np.transpose(dmdt, (2, 0, 1))
        dff.drop(columns='dmdt', inplace=True)

        # apply norms
        for feature, norm in norms.items():
            if not feature in dff.columns: continue
            dff[feature] /= norm
        dff.fillna(0, inplace=True)

        print(modelFile)
        pred = model.predict([dff.values, dmdt], verbose=False).flatten()

    return pred
