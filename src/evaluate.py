import numpy as np
from scipy.stats import rankdata, iqr
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import json
def get_err_median_and_iqr(predicted, groundtruth):
    """
    Calculates the median and interquartile range (IQR) of absolute errors between predicted and ground truth values.

    These statistical measures are used to characterize the distribution of errors, which can help in 
    identifying anomalies or outliers in model predictions.

    Parameters:
    - predicted (array-like): Predicted values from the model.
    - groundtruth (array-like): Actual true values corresponding to the predictions.

    Returns:
    - tuple: A tuple containing:
        - err_median (float): Median of the absolute errors.
        - err_iqr (float): Interquartile range (IQR) of the absolute errors.
    """
    # Compute absolute differences between predicted and ground truth values
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    # Calculate median error
    err_median = np.median(np_arr)
    
    # Calculate interquartile range (IQR) of the errors
    err_iqr = iqr(np_arr)

    return err_median, err_iqr
def get_err_gdn_tra(test_res, val_res):
    """
    Computes the absolute error between predicted values and ground truth values.

    This function is typically used during model evaluation to quantify the difference
    between the model's predictions and the actual true values. It returns the absolute 
    differences (errors) for further analysis or scoring.

    Parameters:
    - test_res (tuple): A tuple containing prediction results and corresponding ground truth.
        - test_predict: Predicted values from the model.
        - test_gt: Actual true values (ground truth).
    - val_res: Not used in this version of the function; may be reserved for future extensions.

    Returns:
    - test_delta (np.ndarray): Array of absolute errors between predictions and ground truth.
    """

    # Unpack test predictions and ground truth
    test_predict, test_gt = test_res

    # Calculate absolute difference between predicted and actual values
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),  # Convert predictions to float64
        np.array(test_gt).astype(np.float64)         # Convert ground truth to float64
    ))

    return test_delta

def get_full_err_scores_tra(test_result, val_result):
    """
    Computes error scores for all features by comparing test predictions with ground truth,
    and also computes normal error distributions using validation data.

    This function iterates over each feature dimension, calculates error scores using 
    [get_err_gdn_tra], and aggregates the results 
    across all features. It is used in model evaluation to assess performance per feature 
    and overall.

    Parameters:
    - test_result (array-like): Test prediction results with shape (time_steps, num_samples, num_features).
    - val_result (array-like): Validation prediction results, typically used as a reference 
                               for normal error distribution.

    Returns:
    - tuple:
        - all_scores (np.ndarray): Aggregated error scores for test data across all features.
        - all_normals (np.ndarray): Aggregated error scores for validation data (normal distribution).
    """

    # Convert input to NumPy arrays if not already done
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    # Initialize containers for aggregated error scores
    all_scores = None
    all_normals = None

    # Determine number of features from the last dimension of test result
    feature_num = np_test_result.shape[-1]

    # Iterate through each feature dimension
    for i in range(feature_num):
        # Extract predicted values and ground truth for current feature from test and val results
        test_re_list = np_test_result[1:, :, i]  # Skip first time step if needed
        val_re_list = np_val_result[1:, :, i]

        # Compute error scores for test data
        scores = get_err_gdn_tra(test_re_list, test_re_list)  # Predicted vs GT
        # Compute error distribution using validation data (self-comparison)
        normal_dist = get_err_gdn_tra(val_re_list, val_re_list)  # Val predicted vs Val GT

        # Stack results vertically (feature-wise)
        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((all_scores, scores))
            all_normals = np.vstack((all_normals, normal_dist))

    return all_scores, all_normals

def search_optimal_threshold(scores, labels):
    """
    Searches for the optimal threshold on error scores that maximizes the F1 score.

    This function performs a binary search to find the best threshold value 
    that separates normal and anomalous data points based on model prediction errors.
    It calculates precision, recall, F1-score, accuracy, and AUC-ROC for the optimal threshold.

    Parameters:
    - scores (np.ndarray): Anomaly scores predicted by the model (higher means more anomalous).
    - labels (np.ndarray): Ground truth labels (0 for normal, 1 for anomaly).

    Returns:
    - best_threshold (float): Threshold value that yields the best F1 score.
    - best_metrics (dict): Dictionary containing evaluation metrics at the best threshold:
        - 'precision': Precision score.
        - 'recall': Recall score.
        - 'f1': F1 score.
        - 'accuracy': Accuracy score.
        - 'roc_auc': ROC AUC score.
        - 'predictions': Binary predictions generated using the best threshold.
    """

    # Rank the scores to reduce sensitivity to absolute values
    scores = rankdata(scores, method='ordinal')

    # Initialize best metrics
    best_f1 = 0
    best_threshold = 0
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'roc_auc': 0.0,
        'predictions': np.array([])
    }

    # Compute class weights to handle imbalance
    weight = np.bincount(labels) / len(labels)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in labels]

    # Set initial search bounds
    low, high = scores.min(), scores.max()
    eps = 1e-5  # Small epsilon for convergence

    # Binary search for best threshold
    while high - low > eps:
        mid = (low + high) / 2
        predictions = (scores >= mid).astype(int)

        # Evaluate current threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', sample_weight=sample_weight
        )

        # Update best metrics if current F1 is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = mid
            best_metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy_score(labels, predictions),
                'roc_auc': roc_auc_score(labels, scores, sample_weight=sample_weight),
                'predictions': predictions
            })

        # Adjust search bounds based on performance
        if f1 > best_f1:
            low, high = mid, high
        else:
            low, high = low, mid

    return best_threshold, best_metrics


def get_best_performance_data(total_err_scores, gt_labels, val_scores, v_gt_labels, topk=1):
    """
    Computes the best performance metrics using top-k error scores.

    This function selects the top-k features with the highest validation errors,
    aggregates their scores, and searches for an optimal threshold to classify anomalies.
    It then computes precision, recall, F1-score, accuracy, and AUC-ROC based on this threshold.

    Parameters:
    - total_err_scores (np.ndarray): Error scores from test data for all features.
    - gt_labels (np.ndarray): Ground truth labels for test data.
    - val_scores (np.ndarray): Error scores from validation data for all features.
    - v_gt_labels (np.ndarray): Ground truth labels for validation data.
    - topk (int): Number of top features to select based on validation error magnitude.

    Returns:
    - best_metrics (dict): Dictionary containing evaluation metrics:
        - 'precision': Precision score.
        - 'recall': Recall score.
        - 'f1': F1 score.
        - 'accuracy': Accuracy score.
        - 'roc_auc': ROC AUC score.
        - 'threshold': Optimal threshold used for classification.
        - 'sigma_label': Final binary ground truth labels.
        - 'y_pred': Predicted labels using optimal threshold.
        - 'scores_search': Aggregated top-k error scores used for threshold search.
        - 'scores_diff': Average error scores across time steps.
    """

    # Determine number of features from validation scores
    total_features = val_scores.shape[0]

    # Find indices of top-k features with highest validation scores
    topk_indices = np.argpartition(val_scores, 
                                    range(total_features - topk - 1, total_features), 
                                    axis=0)[-topk:]

    # Aggregate error scores for top-k features
    total_topk_err_scores = np.sum(np.take_along_axis(val_scores, topk_indices, axis=0), axis=0)

    # Generate final binary labels from ground truth (1 if any anomaly present, else 0)
    labelsFinal = (np.sum(v_gt_labels, axis=0) >= 1) + 0

    # Search for optimal threshold based on validation scores
    threshold, result = search_optimal_threshold(total_topk_err_scores, labelsFinal)

    # Generate predicted labels using found threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    # Convert labels to integers explicitly
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        labelsFinal[i] = int(labelsFinal[i])

    # Recalculate final labels
    labelsFinal = (np.sum(v_gt_labels, axis=0) >= 1) + 0

    # Prepare dictionary of best performance metrics
    best_metrics = {
        'precision': result["precision"],
        'recall': result["recall"],
        'f1': result["f1"],
        'accuracy': result["accuracy"],
        'roc_auc': result["roc_auc"],
        'threshold': threshold,
        'sigma_label': labelsFinal.tolist(),
        'y_pred': np.asarray(result["predictions"]),
        'gt_labels': labelsFinal.tolist(),
        'scores_search': total_topk_err_scores,
        'thresold': threshold,
        "scores_diff": np.mean(total_err_scores.T, axis=1)
    }

    return best_metrics
def get_val_res(scores, labels, th, path=None, search_res=None, test_data=None, val_data=None):
    """
    Evaluates model performance on validation data and returns key evaluation metrics.

    This function computes error scores using [get_full_err_scores_tra],
    then retrieves the best performance metrics using 
    [get_best_performance_data]. It aggregates metrics like precision,
    recall, F1-score, accuracy, and ROC-AUC for final reporting or further analysis.

    Parameters:
    - scores (array-like): Anomaly scores predicted by the model.
    - labels (array-like): Ground truth labels (0 for normal, 1 for anomaly).
    - th (float): Threshold used for binary classification of anomalies.
    - path (str, optional): File path to save results; currently unused in this version.
    - search_res (dict, optional): Previously computed search results; currently unused.
    - test_data (tuple or array-like): Test dataset containing ground truth and predictions.
    - val_data (tuple or array-like): Validation dataset used for threshold tuning.

    Returns:
    - tuple:
        - threshold (float): Optimal threshold found during evaluation.
        - best_metrics (dict): Dictionary containing performance metrics:
            - 'precision': Precision score.
            - 'recall': Recall score.
            - 'f1': F1 score.
            - 'accuracy': Accuracy score.
            - 'roc_auc': ROC AUC score.
            - 'y_pred': Predicted labels.
            - 'gt_labels': Ground truth labels.
            - 'scores_search': Aggregated top-k error scores.
            - 'scores_diff': Average error across time steps.
        - None: Placeholder for additional return values (currently unused).
    """

    # Initialize dictionary to store best performance metrics
    best_metrics = {}

    # Compute error scores for test and validation data
    test_scores, normal_scores = get_full_err_scores_tra(test_data, val_data)

    # Get best performance metrics using validation-based thresholding
    info = get_best_performance_data(
        total_err_scores=test_scores,
        gt_labels=test_data[0].T,         # Extract ground truth from test data
        val_scores=normal_scores,
        v_gt_labels=val_data[0].T,        # Extract ground truth from validation data
        topk=1                            # Use top-1 feature for final scoring
    )

    # Copy relevant metrics into best_metrics dictionary
    best_metrics["precision"] = info["precision"]
    best_metrics["recall"] = info["recall"]
    best_metrics["roc_auc"] = info["roc_auc"]
    best_metrics["f1"] = info["f1"]
    best_metrics["accuracy"] = info["accuracy"]
    best_metrics["predictions"] = info["y_pred"]
    best_metrics["sigma_label"] = info["sigma_label"]
    best_metrics["scores_search"] = info["scores_search"]
    best_metrics["scores_diff"] = info["scores_diff"]
    best_metrics["y_pred"] = info["y_pred"]
    best_metrics["gt_labels"] = info["gt_labels"]

    # Return optimal threshold, performance metrics, and placeholder
    return info["thresold"], best_metrics, None
