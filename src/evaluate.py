import numpy as np
from scipy.stats import iqr, rankdata
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc

def get_err_scores(test_predict, test_gt, train=""):

    n_err_mid, n_err_iqr = np.median(test_predict), iqr(test_predict)

    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)


    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3

    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    return err_scores


def get_val_res(scores, labels, th):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'roc_auc': roc_auc
    }
    precision, recall, f1, accuracy, predictions = calculate_metrics(scores, labels, th)
    best_metrics["precision"] = precision
    best_metrics["recall"] = recall
    best_metrics["roc_auc"] = roc_auc_score(labels, predictions, average="micro")
    best_metrics["f1"] = f1
    best_metrics["accuracy"] = accuracy
    # best_metrics["predictions"] = predictions
    return th, best_metrics


def calculate_metrics(scores, labels, threshold):
    """
    """
    predictions = [1 if score >= threshold else 0 for score in scores]

    # 计算精确率、召回率和F1分数
    true_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    false_negatives = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    true_negatives = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(labels)

    return precision, recall, f1, accuracy, predictions


def search_optimal_threshold(scores, labels):

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    scores_sorted = rankdata(scores, method='ordinal')
    best_f1 = 0
    best_threshold = 0
    if np.isnan(roc_auc):
        roc_auc = 0.0
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'roc_auc': roc_auc
    }

    for i in scores_sorted:
        threshold = scores[i - 1]
        precision, recall, f1, accuracy, predictions = calculate_metrics(scores, labels, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics["precision"] = precision
            best_metrics["recall"] = recall
            best_metrics["f1"] = f1
            best_metrics["accuracy"] = accuracy
            best_metrics["predictions"] = predictions

    return best_threshold, best_metrics
