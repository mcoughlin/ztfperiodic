
def classify(algorithm, features, modelFile=None, configFile=None):

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
        import yaml

        model = load_model(modelFile)

        with open(configFile) as config_yaml:
            config = yaml.load(config_yaml, Loader=yaml.FullLoader)

        dff = deepcopy(features)
        dmdt = np.expand_dims(np.array([d for d in dff['dmdt'].values]),
                              axis=-1)
        dff.drop(columns='dmdt', inplace=True)

        model_class = modelFile.split("/")[-1].split("-")[0]

        # scale features
        train_config = config["training"]["classes"][model_class]
        feature_names = config["features"][train_config["features"]]
        feature_stats = config.get("feature_stats", None)
        scale_features = "min_max"
    
        for feature in feature_names:
            stats = feature_stats.get(feature)
            if (stats is not None) and (stats["std"] != 0):
                if scale_features == "median_std":
                    dff[feature] = (dff[feature] - stats["median"]) / stats["std"]
                elif scale_features == "min_max":
                    dff[feature] = (dff[feature] - stats["min"]) / (
                        stats["max"] - stats["min"]
                    )

        dff.fillna(0, inplace=True)
        pred = model.predict([dff.values, dmdt], verbose=False).flatten()

    return pred
