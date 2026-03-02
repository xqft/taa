from comet_ml import Experiment
import importlib
import utils.HiggsBosonCompetition_AMSMetric_rev1 as hb_metrics

importlib.reload(hb_metrics)

import numpy as np
from utils.HiggsBosonCompetition_AMSMetric_rev1 import ams_scorer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from utils.metrics import MetricsManager

def exp_eval_nn(model_name, model, X_test, y_test, sample_weight, api_key):
    experiment = Experiment(
        api_key=api_key,
        project_name="proyecto-1-taa",
        workspace="xqft"
    )
    experiment.set_name("Evaluate " + model_name)
    experiment.add_tags(["Evaluation", model_name])

    y_pred_proba = np.array(model.predict(X_test)).flatten()
    y_true = y_test
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    threshold = MetricsManager.get_best_threshold(precision, recall, thresholds)
    y_pred = np.array((y_pred_proba >= threshold).astype(int)).flatten()

    experiment.log_metric("AMS", ams_scorer(y_true, y_pred, sample_weight))

    experiment.log_curve("Precision-Recall", recall[:-1], precision[:-1])
    experiment.log_curve("Threshold-Precision", thresholds.tolist(), precision[:-1])
    experiment.log_curve("Threshold-Recall", thresholds.tolist(), recall[:-1])
    experiment.log_curve("ROC", fpr, tpr)
    
    experiment.end()

def exp_eval_classifier(model_name, model, X_test, y_test, sample_weight, api_key):
    experiment = Experiment(
        api_key=api_key,
        project_name="proyecto-1-taa",
        workspace="xqft"
    )
    experiment.set_name("Evaluate " + model_name)
    experiment.add_tags(["Evaluation", model_name])

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 0]
    y_true = y_test

    experiment.log_metric("AMS", ams_scorer(y_true, y_pred, sample_weight))

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    experiment.log_curve("Precision-Recall", recall[:-1], precision[:-1])
    experiment.log_curve("Threshold-Precision", thresholds, precision[:-1])
    experiment.log_curve("Threshold-Recall", thresholds, recall[:-1])
    experiment.log_curve("ROC", tpr, fpr)
    
    experiment.end()


def exp_tune_classifier(model_name, search, api_key):
    for i in range(len(search.cv_results_['params'])):
        experiment = Experiment(
            api_key=api_key,
            project_name="proyecto-1-taa",
            workspace="xqft"
        )
        experiment.add_tags(["Hyperparameter tuning", model_name, f"iter {i}"])
        experiment.set_name("Tune " + model_name + f" (iter {i})")

        for k, v in search.cv_results_.items():
            if k == "params":
                experiment.log_parameters(v[i])
            else:
                experiment.log_metric(k, v[i])
        experiment.end()
