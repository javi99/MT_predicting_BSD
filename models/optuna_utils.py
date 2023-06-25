"""
Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and its
hyperparameters.

"""

import numpy as np
import optuna
import xgboost as xgb
import sklearn.metrics


class Optuna:

    def __init__(self, training, validation, test):
        self.training_data = training
        self.validation_data = validation
        self.test_data = test

    def xgb_objective(self, trial):

        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            # use approx for large datasets
            "tree_method": "approx",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            #"subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


        #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "rmse")
        bst = xgb.train(param, dtrain = self.training_data)
        preds = bst.predict(self.test_data)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.test_data.get_label(), pred_labels)
        
        return accuracy

    def conduct_study(self, model):
        if model == 'xgb':
            modeller = self.xgb_objective
        study = optuna.create_study(direction="maximize")
        study.optimize(modeller, n_trials=100, timeout=600)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return trial.params
