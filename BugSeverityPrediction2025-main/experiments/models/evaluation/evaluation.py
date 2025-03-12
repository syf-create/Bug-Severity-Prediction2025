import os.path
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve, confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

labels = {0: "Critical, Blocker", 1: "Major, High", 2: "Medium", 3: "Low, Trivial, Minor"}
colors = cycle(["#C0392B", "#E67E22", "#F1C40F", "#71B5A0"])
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def evaluatelog_result(y_true, y_prediction, prob, logger):
    # logger.info("************ " + model_name + " ************")
    eval_result = evaluate_result(y_true=y_true, y_prediction=y_prediction, prob=prob)
    # plot_roc(y_test=y_true, prob=prob, model_name=model_name)
    # plot_precision_recall(y_test=y_true, prob=prob, model_name=model_name)
    # plot_confusionmatrix(y_true, y_prediction, model_name=model_name)

    for key in sorted(eval_result.keys()):
        logger.info("  %s = %s", key, str(eval_result[key]))

    # 返回所有评估指标，便于在test函数中使用
    return (
        eval_result["eval_f1"],
        eval_result["eval_f1_perclass"],
        eval_result["eval_acc"],
        eval_result["eval_precision"],
        eval_result["eval_recall"],
        eval_result["eval_ROC-UAC"],
        eval_result["eval_mcc"]
    )

def evaluate_result(y_true, y_prediction, prob):
    f1_weighted = f1_score(y_true, y_prediction, average='weighted')
    f1_per_class = f1_score(y_true, y_prediction, average=None)
    accuracy = accuracy_score(y_true, y_prediction)
    precision = precision_score(y_true, y_prediction, average='weighted')
    recall = recall_score(y_true, y_prediction, average='weighted')
    roc_uac = roc_auc_score(y_true, prob, average='weighted', multi_class='ovo')
    mcc = matthews_corrcoef(y_true, y_prediction)

    eval_result = {
        "eval_f1": float(f1_weighted),
        "eval_f1_perclass": f1_per_class,
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_ROC-UAC": float(roc_uac),
        "eval_mcc": float(mcc)
    }

    return eval_result

