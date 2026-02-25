import torch
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import json

from src.pot import adjust_predicts

def get_attack_interval(attack):
    """
    Identify the start and end indices of continuous attack intervals in a binary attack signal.

    Args:
        attack: A list or array of binary values (0 or 1), where 1 indicates an attack.

    Returns:
        res: A list of tuples, where each tuple contains the start and end indices of an attack interval.
    """
    # Initialize lists to store the start (heads) and end (tails) indices of attack intervals
    heads = []
    tails = []

    # Iterate through the attack signal to detect transitions
    for i in range(len(attack)):
        # Check if the current position is part of an attack (value is 1)
        if attack[i] == 1:
            # If the previous position was not part of an attack (value is 0), mark the start of a new interval
            if attack[i - 1] == 0:
                heads.append(i)

            # If the next position is not part of an attack or we are at the end of the signal, mark the end of the interval
            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)

    # Combine the start and end indices into tuples representing attack intervals
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))

    # Return the list of attack intervals
    return res



def eval_scores(scores, true_scores, th_steps, return_thresold=True, average="macro"):
    """
    Evaluate F1 scores across a range of thresholds for anomaly detection.

    Args:
        scores: Anomaly scores or error scores for each sample.
        true_scores: Ground truth labels for the samples.
        th_steps: Number of thresholds to evaluate.
        return_thresold: Boolean flag to determine whether to return thresholds along with F1 scores (default is True).
        average: Type of averaging for F1 score calculation (default is "macro").

    Returns:
        fmeas: List of F1 scores computed for each threshold.
        thresholds: List of thresholds used for evaluation (returned only if `return_thresold` is True).
    """
    # Pad the scores list with zeros if it is shorter than the true_scores list
    padding_list = [0] * (len(true_scores) - len(scores))
    if len(padding_list) > 0:
        scores = padding_list + scores

    # Compute class weights for imbalanced datasets
    weight = np.bincount(true_scores) / len(true_scores)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in true_scores]

    # Rank the scores ordinally to facilitate threshold-based predictions
    scores_sorted = rankdata(scores, method='ordinal')

    # Define the range of thresholds to evaluate
    th_vals = np.linspace(np.min(scores), np.max(scores), th_steps)

    # Initialize lists to store F1 scores and corresponding thresholds
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps

    # Evaluate F1 scores for each threshold
    for i in range(th_steps):
        # Generate binary predictions based on the current threshold
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        # Compute the F1 score for the current predictions
        try:
            fmeas[i] = f1_score(true_scores, cur_pred, average=average, sample_weight=sample_weight)
        except:
            fmeas[i] = 0.0  # Default to 0.0 if F1 score calculation fails

        # Store the current threshold
        thresholds[i] = th_vals[i]

    # Return F1 scores and thresholds (if requested)
    if return_thresold:
        return fmeas, thresholds
    return fmeas


def eval_mseloss(predicted, ground_truth):
    """
    Compute the Mean Squared Error (MSE) loss between predicted and ground truth values.

    Args:
        predicted: Predicted values from the model.
        ground_truth: Ground truth values.

    Returns:
        loss: The computed MSE loss.
    """
    # Convert inputs to NumPy arrays for consistent processing
    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    # Compute the Mean Squared Error (MSE) loss using sklearn's mean_squared_error function
    loss = mean_squared_error(predicted_list, ground_truth_list)

    # Return the computed loss
    return loss


def get_err_median_and_iqr(predicted, groundtruth):
    """
    Compute the median and interquartile range (IQR) of the absolute differences 
    between predicted and ground truth values.

    Args:
        predicted: Predicted values from the model.
        groundtruth: Ground truth values.

    Returns:
        err_median: Median of the absolute differences.
        err_iqr: Interquartile range (IQR) of the absolute differences.
    """
    # Compute the absolute differences between predicted and ground truth values
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    # Calculate the median of the absolute differences
    err_median = np.median(np_arr)

    # Calculate the interquartile range (IQR) of the absolute differences
    err_iqr = iqr(np_arr)

    # Return the median and IQR
    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):
    """
    Compute the median and quantile-based delta (difference between upper and lower quantiles)
    of the absolute differences between predicted and ground truth values.

    Args:
        predicted: Predicted values from the model.
        groundtruth: Ground truth values.
        percentage: Percentage value to define the quantile range (e.g., 0.9 for 90th percentile).

    Returns:
        err_median: Median of the absolute differences.
        err_delta: Difference between the upper and lower quantiles (quantile delta).
    """
    # Compute the absolute differences between predicted and ground truth values
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    # Calculate the median of the absolute differences
    err_median = np.median(np_arr)

    # Calculate the quantile delta: difference between the upper and lower quantiles
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    # Return the median and quantile delta
    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):
    """
    Compute the trimmed mean and quantile-based delta (difference between upper and lower quantiles)
    of the absolute differences between predicted and ground truth values.

    Args:
        predicted: Predicted values from the model.
        groundtruth: Ground truth values.
        percentage: Percentage value to define the quantile range (e.g., 0.9 for 90th percentile).

    Returns:
        err_median: Trimmed mean of the absolute differences.
        err_delta: Difference between the upper and lower quantiles (quantile delta).
    """
    # Compute the absolute differences between predicted and ground truth values
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    # Calculate the trimmed mean of the absolute differences
    err_median = trim_mean(np_arr, percentage)

    # Calculate the quantile delta: difference between the upper and lower quantiles
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    # Return the trimmed mean and quantile delta
    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):
    """
    Compute the mean and standard deviation of the absolute differences 
    between predicted and ground truth values.

    Args:
        predicted: Predicted values from the model.
        groundtruth: Ground truth values.

    Returns:
        err_mean: Mean of the absolute differences.
        err_std: Standard deviation of the absolute differences.
    """
    # Compute the absolute differences between predicted and ground truth values
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    # Calculate the mean of the absolute differences
    err_mean = np.mean(np_arr)

    # Calculate the standard deviation of the absolute differences
    err_std = np.std(np_arr)

    # Return the mean and standard deviation
    return err_mean, err_std


def get_f1_score(scores, gt, contamination):
    """
    Compute the F1 score for anomaly detection based on a contamination-based threshold.

    Args:
        scores: Anomaly scores or error scores for each sample.
        gt: Ground truth labels for the samples.
        contamination: Proportion of expected anomalies in the data (used to determine the threshold).

    Returns:
        f1: F1 score computed using sklearn.
    """
    # Create a padding list to align the lengths of scores and ground truth labels
    padding_list = [0] * (len(gt) - len(scores))

    # Determine the threshold based on the contamination level (percentile-based)
    threshold = percentile(scores, 100 * (1 - contamination))

    # Pad the scores list if it is shorter than the ground truth labels
    if len(padding_list) > 0:
        scores = padding_list + scores

    # Generate binary predictions based on the threshold
    pred_labels = (scores > threshold).astype('int').ravel()

    # Compute and return the F1 score using sklearn
    return f1_score(gt, pred_labels)

def get_full_err_scores(test_result, val_result):
    """
    Calculate error scores for test and validation results across all features.
    
    Args:
        test_result: Test result data, typically containing predictions and ground truth.
        val_result: Validation result data, used for normalization.
        
    Returns:
        all_scores: Error scores for the test data.
        all_normals: Normalized error scores based on validation data.
    """
    # Convert input data to NumPy arrays for easier manipulation
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    # Initialize variables to store aggregated scores and normalized distributions
    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]  # Number of features in the data

    # Extract labels from the test result (assumed to be in the third row of the first column)
    labels = np_test_result[2, :, 0].tolist()

    # Iterate over each feature to compute error scores
    for i in range(feature_num):
        # Extract prediction and ground truth for the current feature
        test_re_list = np_test_result[:2, :, i]  # First two rows: predictions and ground truth
        val_re_list = np_val_result[:2, :, i]    # Same for validation data

        # Compute error scores for test and validation data
        scores = get_err_gdn(test_re_list, val_re_list)       # Error scores for test data
        normal_dist = get_err_gdn(val_re_list, val_re_list)   # Normalized distribution from validation data

        # Stack scores and normalized distributions vertically
        if all_scores is None:
            # Initialize with the first feature's scores
            all_scores = scores
            all_normals = normal_dist
        else:
            # Append subsequent features' scores
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    # Return aggregated error scores and normalized distributions
    return all_scores, all_normals

def get_full_err_scores_tra(test_result, val_result):
    """
    Calculate error scores for test and validation results across all features,
    specifically tailored for training-related computations.

    Args:
        test_result: Test result data, typically containing predictions and ground truth.
        val_result: Validation result data, used for normalization.

    Returns:
        all_scores: Error scores for the test data.
        all_normals: Normalized error scores based on validation data.
    """
    # Convert input data to NumPy arrays for easier manipulation
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    # Initialize variables to store aggregated scores and normalized distributions
    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]  # Number of features in the data

    # Note: Labels extraction is commented out here but may be needed in other contexts
    # labels = np_test_result[2, :, 0].tolist()

    # Iterate over each feature to compute error scores
    for i in range(feature_num):
        # Extract prediction and ground truth for the current feature
        test_re_list = np_test_result[:2, :, i]  # First two rows: predictions and ground truth
        val_re_list = np_val_result[:2, :, i]    # Same for validation data

        # Compute error scores for test and validation data using training-specific logic
        scores = get_err_gdn_tra(test_re_list, val_re_list)       # Error scores for test data
        normal_dist = get_err_gdn_tra(val_re_list, val_re_list)   # Normalized distribution from validation data

        # Stack scores and normalized distributions vertically
        if all_scores is None:
            # Initialize with the first feature's scores
            all_scores = scores
            all_normals = normal_dist
        else:
            # Append subsequent features' scores
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    # Return aggregated error scores and normalized distributions
    return all_scores, all_normals


def tensor_to_numpy(obj):
    """
    Convert an object (potentially a PyTorch tensor, list of tensors, or nested structures)
    into NumPy format, automatically handling GPU-to-CPU transfer.

    Args:
        obj: An object that may contain PyTorch tensors.

    Returns:
        The converted object in NumPy format.
    """
    # If the object is a PyTorch tensor
    if isinstance(obj, torch.Tensor):
        # First move it to CPU, detach from the computation graph, then convert to NumPy
        return obj.detach().cpu().numpy()

    # If the object is a list or tuple, recursively process each element
    if isinstance(obj, (list, tuple)):
        return [tensor_to_numpy(item) for item in obj]

    # If the object is a dictionary, recursively process each value
    if isinstance(obj, dict):
        return {key: tensor_to_numpy(value) for key, value in obj.items()}

    # For other types, attempt direct conversion to NumPy array
    try:
        return np.array(obj)
    except:
        # If conversion fails, return the object as-is
        return obj

def get_full_err_scores_ori(test_result, val_result):
    """
    Calculate error scores for test and validation results across all features,
    using the original method for score computation.

    Args:
        test_result: Test result data, typically containing predictions and ground truth.
        val_result: Validation result data, used for normalization.

    Returns:
        all_scores: Error scores for the test data.
        all_normals: Normalized error scores based on validation data.
    """
    # Convert input data to NumPy arrays after transforming tensors to NumPy format
    np_test_result = np.asarray(tensor_to_numpy(test_result))
    np_val_result = np.asarray(tensor_to_numpy(val_result))

    # Initialize variables to store aggregated scores and normalized distributions
    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]  # Number of features in the data

    # Iterate over each feature to compute error scores
    for i in range(feature_num):
        # Extract prediction and ground truth for the current feature
        test_re_list = np_test_result[:2, :, i]  # First two rows: predictions and ground truth
        val_re_list = np_val_result[:2, :, i]    # Same for validation data

        # Compute error scores for test and validation data using the original method
        scores = get_err_gdn_ori(test_re_list, val_re_list)       # Error scores for test data
        normal_dist = get_err_gdn_ori(val_re_list, val_re_list)   # Normalized distribution from validation data

        # Stack scores and normalized distributions vertically
        if all_scores is None:
            # Initialize with the first feature's scores
            all_scores = scores
            all_normals = normal_dist
        else:
            # Append subsequent features' scores
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    # Return aggregated error scores and normalized distributions
    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    """
    Compute the final error scores by taking the maximum error score across all features
    for each time step.

    Args:
        test_result: Test result data, typically containing predictions and ground truth.
        val_result: Validation result data, used for normalization.

    Returns:
        all_scores: Final error scores, representing the maximum error across features for each time step.
    """
    # Retrieve full error scores and normalized distributions from the test and validation results
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    # Take the maximum error score across all features for each time step
    all_scores = np.max(full_scores, axis=0)

    # Return the final aggregated error scores
    return all_scores


def get_smooth_scores(data):
    """
    Compute smoothed error scores by normalizing the data and applying a moving average filter.

    Args:
        data: Input data array, typically representing raw error scores.

    Returns:
        err_scores: Normalized error scores without smoothing applied.
    """
    # Replace NaN values with 0.0 (commented out in the original code)
    # data = np.nan_to_num(data, nan=0.0)

    # Compute the median and interquartile range (IQR) of the data
    n_err_mid, n_err_iqr = np.median(data), iqr(data)
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the data using median and IQR
    err_scores = (data - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Initialize an array to store smoothed scores
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3  # Window size for smoothing

    # Apply a moving average filter to smooth the scores
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    # Return the normalized (but not smoothed) error scores
    return err_scores


def get_err_scores(test_predict, test_gt, train=""):
    """
    Compute error scores between predicted and ground truth values, 
    normalize them, and apply smoothing.

    Args:
        test_predict: Predicted values from the model.
        test_gt: Ground truth values.
        train: Optional parameter for training-specific logic (not used here).

    Returns:
        smoothed_err_scores: Smoothed error scores after normalization.
    """
    # Replace NaN values with 0.0 (commented out in the original code)
    # test_predict = np.nan_to_num(test_predict, nan=0.0)

    # Compute the median and interquartile range (IQR) of the predicted values
    # n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)  # Alternative method (commented out)
    n_err_mid, n_err_iqr = np.median(test_predict), iqr(test_predict)

    # Compute the absolute difference between predicted and ground truth values
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the error scores using median and IQR
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Initialize an array to store smoothed error scores
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3  # Window size for smoothing

    # Apply a moving average filter to smooth the error scores
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    # Return the smoothed error scores
    return smoothed_err_scores


def get_err_gdn(test_res, val_res):
    """
    Compute error scores for test data using median and IQR normalization,
    and apply smoothing to the results.

    Args:
        test_res: Tuple containing test predictions and ground truth.
        val_res: Tuple containing validation predictions and ground truth (not used here).

    Returns:
        test_delta: Absolute difference between test predictions and ground truth.
    """
    # Unpack test predictions and ground truth
    test_predict, test_gt = test_res
    # val_predict, val_gt = val_res  # Not used in this function

    # Compute median and interquartile range (IQR) for normalization
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    # Calculate the absolute difference between predictions and ground truth
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the error scores using median and IQR
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Initialize an array to store smoothed error scores
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3  # Window size for smoothing

    # Apply a moving average filter to smooth the error scores
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    # Return the absolute difference (unsmoothed error scores)
    return test_delta
def get_err_gdn_tra(test_res, val_res):
    """
    Compute error scores for test data using median and IQR normalization,
    specifically tailored for training-related computations.

    Args:
        test_res: Tuple containing test predictions and ground truth.
        val_res: Tuple containing validation predictions and ground truth.

    Returns:
        test_delta: Absolute difference between test predictions and ground truth.
    """
    # Unpack test and validation predictions and ground truth
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    # Compute median and interquartile range (IQR) for normalization
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    # Calculate the absolute difference between test predictions and ground truth
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the error scores using median and IQR
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Initialize an array to store smoothed error scores
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3  # Window size for smoothing

    # Apply a moving average filter to smooth the error scores
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    # Return the absolute difference (unsmoothed error scores)
    return test_delta
def smooth_data(data, windows=5):
    """
    Smooth the input data using a moving average filter after normalizing it
    with median and interquartile range (IQR).

    Args:
        data: Input data array to be smoothed.
        windows: Size of the moving average window (default is 5).

    Returns:
        smoothed_err_scores: Smoothed version of the input data.
    """
    # Initialize an array to store the smoothed scores
    smoothed_err_scores = np.zeros(data.shape)

    # Compute the median and interquartile range (IQR) of the data
    n_err_mid = np.median(data)
    n_err_iqr = iqr(data)
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the data using median and IQR
    data = (data - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Apply a moving average filter to smooth the data
    for i in range(windows, len(data)):
        smoothed_err_scores[i] = np.mean(data[i - windows:i + 1])

    # Return the smoothed data
    return smoothed_err_scores


def get_err_gdn_ori(test_res, val_res):
    """
    Compute error scores for test data using median and IQR normalization,
    and apply smoothing to the results. This is the original method for score computation.

    Args:
        test_res: Tuple containing test predictions and ground truth.
        val_res: Tuple containing validation predictions and ground truth.

    Returns:
        smoothed_err_scores: Smoothed error scores after normalization.
    """
    # Unpack test and validation predictions and ground truth
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    # Compute median and interquartile range (IQR) for normalization
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    # Calculate the absolute difference between test predictions and ground truth
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2  # Small constant to avoid division by zero

    # Normalize the error scores using median and IQR
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    # Initialize an array to store smoothed error scores
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3  # Window size for smoothing

    # Apply a moving average filter to smooth the error scores
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    # Return the smoothed error scores
    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    """
    Compute F1 scores by selecting the top-k error scores for each time step
    and evaluating them against ground truth labels.

    Args:
        total_err_scores: A 2D array of error scores for all features and time steps.
        gt_labels: Ground truth labels for anomaly detection.
        topk: Number of top error scores to consider for each time step (default is 1).

    Returns:
        final_topk_fmeas: F1 scores computed from the top-k error scores.
    """
    # Get the total number of features
    total_features = total_err_scores.shape[0]

    # Find the indices of the top-k error scores for each time step
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Transpose the indices for easier iteration
    topk_indices = np.transpose(topk_indices)

    # Initialize lists to store aggregated top-k scores and mappings
    total_topk_err_scores = []
    topk_err_score_map = []

    # Aggregate the top-k error scores for each time step
    for i, indexs in enumerate(topk_indices):
        # Sum the sorted top-k scores for the current time step
        sum_score = sum(
            score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)]))
        )
        total_topk_err_scores.append(sum_score)

    # Evaluate the aggregated top-k scores against ground truth labels
    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    # Return the computed F1 scores
    return final_topk_fmeas


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    """
    Evaluate the performance of anomaly detection using error scores and ground truth labels.

    Args:
        total_err_scores: A 2D array of error scores for all features and time steps.
        normal_scores: Normalized error scores from validation data.
        gt_labels: Ground truth labels for anomaly detection.
        topk: Number of top error scores to consider for each time step (default is 1).

    Returns:
        best_metrics: A dictionary containing evaluation metrics such as precision, recall, F1-score, accuracy, and ROC-AUC.
    """
    # Get the total number of features
    total_features = total_err_scores.shape[0]

    # Find the indices of the top-k error scores for each time step
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Initialize lists to store aggregated top-k scores and mappings
    total_topk_err_scores = []
    topk_err_score_map = []

    # Aggregate the top-k error scores for each time step
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    # Determine the threshold for anomaly detection based on normalized scores
    thresold = np.max(normal_scores)

    # Generate binary predictions based on the threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    # Convert predictions and labels to integers for consistency
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    # Compute standard evaluation metrics using sklearn
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, pred_labels)

    # Manually compute true positives, false positives, false negatives, and true negatives
    true_positives = sum(1 for p, l in zip(gt_labels, pred_labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(gt_labels, pred_labels) if p == 1 and l == 0)
    false_negatives = sum(1 for p, l in zip(gt_labels, pred_labels) if p == 0 and l == 1)
    true_negatives = sum(1 for p, l in zip(gt_labels, pred_labels) if p == 0 and l == 0)

    # Compute precision, recall, F1-score, and accuracy manually
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(pred_labels)

    # Attempt to compute ROC-AUC with micro averaging; fallback to 0.0 if it fails
    try:
        auc_score = roc_auc_score(gt_labels, pred_labels, average='micro')
    except:
        auc_score = 0.0

    # Compute accuracy using sklearn
    acc = accuracy_score(gt_labels, pred_labels)

    # Store all computed metrics in a dictionary
    best_metrics = {
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'accuracy': acc,
        'roc_auc': auc_score,
        'threshold': thresold,
        'y_pred': pred_labels,
        'gt_labels': gt_labels,
        'scores_search': total_topk_err_scores,
    }

    # Return the dictionary of evaluation metrics
    return best_metrics
def get_best_performance_data(total_err_scores, gt_labels, val_scores, v_gt_labels, topk=1):
    """
    Compute the best performance metrics for anomaly detection by optimizing the threshold
    and evaluating predictions against ground truth labels.

    Args:
        total_err_scores: Error scores for the test data.
        gt_labels: Ground truth labels for the test data.
        val_scores: Error scores for the validation data.
        v_gt_labels: Ground truth labels for the validation data.
        topk: Number of top error scores to consider for each time step (default is 1).

    Returns:
        best_metrics: A dictionary containing the best performance metrics including precision, recall, F1-score, accuracy, ROC-AUC, and threshold.
    """
    # Get the total number of features in the validation scores
    total_features = val_scores.shape[0]

    # Identify the indices of the top-k error scores for each time step in the validation data
    topk_indices = np.argpartition(val_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Aggregate the top-k error scores for each time step
    total_topk_err_scores = np.sum(np.take_along_axis(val_scores, topk_indices, axis=0), axis=0)

    # Convert validation ground truth labels to binary format (1 if any feature is anomalous, 0 otherwise)
    labelsFinal = (np.sum(v_gt_labels, axis=0) >= 1) + 0

    # Search for the optimal threshold that maximizes F1-score
    thresold, result = search_optimal_threshold(total_topk_err_scores, labelsFinal)

    # Generate binary predictions based on the optimal threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    # Ensure predictions and labels are integers for consistency
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        labelsFinal[i] = int(labelsFinal[i])

    # Extract performance metrics from the optimization result
    pre = result["precision"]
    rec = result["recall"]
    f1 = result["f1"]
    acc = result["accuracy"]
    auc_score = result["roc_auc"]

    # Re-convert validation ground truth labels to binary format (redundant but ensures consistency)
    labelsFinal = (np.sum(v_gt_labels, axis=0) >= 1) + 0

    # Compile all metrics into a dictionary
    best_metrics = {
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'accuracy': acc,
        'roc_auc': auc_score,
        'threshold': thresold,
        'sigma_label': labelsFinal.tolist(),
        'y_pred': np.asarray(result["predictions"]),
        'gt_labels': labelsFinal.tolist(),
        'scores_search': total_topk_err_scores,
        'thresold': thresold,
        "scores_diff": np.mean(total_err_scores.T, axis=1)
    }

    # Return the dictionary of best performance metrics
    return best_metrics

def testThreshouldPerfermance_pot(test_socres, test_labels, threshold, type="macro"):
    """
    Evaluate the performance of anomaly detection using a given threshold,
    specifically tailored for POT (Peaks Over Threshold) method.

    Args:
        test_socres: Test scores or anomaly scores.
        test_labels: Ground truth labels for the test data.
        threshold: Threshold value for determining anomalies.
        type: Type of averaging for metrics (default is "macro").

    Returns:
        result: A dictionary containing performance metrics such as precision, recall, F1-score, accuracy, ROC-AUC, AP, and latency.
    """
    # Initialize an empty dictionary to store results
    result = {}

    # Adjust predictions and calculate latency using the POT method
    pred_labels, p_latency = adjust_predicts(test_socres, test_labels, threshold, calc_latency=True)

    # Compute class weights for imbalanced datasets
    weight = np.bincount(test_labels) / len(test_labels)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in test_labels]

    # Calculate performance metrics using weighted averages
    precision = precision_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    recall = recall_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    f1 = f1_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    accuracy = accuracy_score(test_labels, pred_labels, sample_weight=sample_weight)
    roc_auc = roc_auc_score(test_labels, test_socres, sample_weight=sample_weight, average=type)
    ap = average_precision_score(test_labels, test_socres, average=type, sample_weight=sample_weight)

    # Store all computed metrics in the result dictionary
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["accuracy"] = accuracy
    result["roc_auc"] = roc_auc
    result["AP"] = ap
    result["threshold"] = threshold
    result["pred"] = pred_labels
    result["p_latency"] = p_latency

    # Return the dictionary of performance metrics
    return result

def testThreshouldPerfermance(test_socres, test_labels, threshold, type="macro"):
    """
    Evaluate the performance of anomaly detection using a given threshold.

    Args:
        test_socres: Test scores or anomaly scores.
        test_labels: Ground truth labels for the test data.
        threshold: Threshold value for determining anomalies.
        type: Type of averaging for metrics (default is "macro").

    Returns:
        result: A dictionary containing performance metrics such as precision, recall, F1-score, accuracy, ROC-AUC, and AP.
    """
    # Initialize an empty dictionary to store results
    result = {}

    # Generate binary predictions based on the threshold
    pred_labels = np.zeros(len(test_labels))
    pred_labels[test_socres > threshold] = 1

    # Compute class weights for imbalanced datasets
    weight = np.bincount(test_labels) / len(test_labels)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in test_labels]

    # Calculate performance metrics using weighted averages
    precision = precision_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    recall = recall_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    f1 = f1_score(test_labels, pred_labels, sample_weight=sample_weight, average=type)
    accuracy = accuracy_score(test_labels, pred_labels, sample_weight=sample_weight)
    roc_auc = roc_auc_score(test_labels, test_socres, sample_weight=sample_weight, average=type)
    ap = average_precision_score(test_labels, test_socres, average=type, sample_weight=sample_weight)

    # Store all computed metrics in the result dictionary
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["accuracy"] = accuracy
    result["roc_auc"] = roc_auc
    result["AP"] = ap
    result["threshold"] = threshold
    result["pred"] = pred_labels

    # Return the dictionary of performance metrics
    return result

def testThreshouldPerfermance_Trandation(socres, gt_labels, thresold, topk=1, average="macro"):
    """
    Evaluate the performance of anomaly detection using a given threshold and top-k scoring,
    specifically tailored for transition-related computations.

    Args:
        socres: Anomaly scores or error scores for each time step and feature.
        gt_labels: Ground truth labels for the data.
        thresold: Threshold value for determining anomalies.
        topk: Number of top error scores to consider for each time step (default is 1).
        average: Type of averaging for metrics (default is "macro").

    Returns:
        best_metrics: A dictionary containing performance metrics such as precision, recall, F1-score, accuracy, ROC-AUC, and AP.
    """
    # Get the total number of features in the scores
    total_features = socres.shape[0]

    # Identify the indices of the top-k error scores for each time step
    topk_indices = np.argpartition(socres, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Aggregate the top-k error scores for each time step
    total_topk_err_scores = np.sum(np.take_along_axis(socres, topk_indices, axis=0), axis=0)

    # Convert ground truth labels to binary format (1 if any feature is anomalous, 0 otherwise)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Generate binary predictions based on the threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    # Ensure predictions and labels are integers for consistency
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        labelsFinal[i] = int(labelsFinal[i])

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labelsFinal) / len(labelsFinal)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in labelsFinal]

    # Calculate performance metrics using weighted averages
    pre = precision_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    rec = recall_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    f1 = f1_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    acc = accuracy_score(labelsFinal, pred_labels, sample_weight=sample_weight)
    auc_score = roc_auc_score(labelsFinal, total_topk_err_scores, sample_weight=sample_weight)
    ap = average_precision_score(labelsFinal, total_topk_err_scores, average=average, sample_weight=sample_weight)

    # Re-convert ground truth labels to binary format (redundant but ensures consistency)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Compile all metrics into a dictionary
    best_metrics = {
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'accuracy': acc,
        'roc_auc': auc_score,
        'threshold': thresold,
        'sigma_label': labelsFinal.tolist(),
        'y_pred': pred_labels.tolist(),
        'AP': ap,
        'gt_labels': labelsFinal.tolist(),
        'scores_search': total_topk_err_scores.tolist(),
        'thresold': thresold,
        "scores_diff": np.mean(socres.T, axis=1)
    }

    # Return the dictionary of performance metrics
    return best_metrics
def get_best_performance_data_ori(scores, gt_labels, topk=1, average="macro"):
    """
    Compute the best performance metrics for anomaly detection by optimizing the threshold
    and evaluating predictions against ground truth labels, using the original method.

    Args:
        scores: Anomaly scores or error scores for each time step and feature.
        gt_labels: Ground truth labels for the data.
        topk: Number of top error scores to consider for each time step (default is 1).
        average: Type of averaging for metrics (default is "macro").

    Returns:
        best_metrics: A dictionary containing the best performance metrics including precision, recall, F1-score, accuracy, ROC-AUC, and threshold.
    """
    # Get the total number of features in the scores
    total_features = scores.shape[0]

    # Identify the indices of the top-k error scores for each time step
    topk_indices = np.argpartition(scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Aggregate the top-k error scores for each time step
    total_topk_err_scores = np.sum(np.take_along_axis(scores, topk_indices, axis=0), axis=0)

    # Convert ground truth labels to binary format (1 if any feature is anomalous, 0 otherwise)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Search for the optimal threshold that maximizes F1-score
    thresold, result = search_optimal_threshold(total_topk_err_scores, labelsFinal, average=average)

    # Generate binary predictions based on the optimal threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labelsFinal) / len(labelsFinal)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in labelsFinal]

    # Calculate performance metrics using weighted averages
    precision = precision_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    recall = recall_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    f1 = f1_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    accuracy = accuracy_score(labelsFinal, pred_labels, sample_weight=sample_weight)
    roc_auc = roc_auc_score(labelsFinal, total_topk_err_scores, sample_weight=sample_weight, average=average)
    ap = average_precision_score(labelsFinal, total_topk_err_scores, average=average, sample_weight=sample_weight)

    # Re-convert ground truth labels to binary format (redundant but ensures consistency)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Compile all metrics into a dictionary
    best_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'threshold': thresold,
        "AP": ap,
        'sigma_label': labelsFinal.tolist(),
        'y_pred': pred_labels.tolist(),
        'gt_labels': labelsFinal.tolist(),
        'scores_search': total_topk_err_scores.tolist(),
        'thresold': thresold,
        "scores_diff": np.mean(scores.T, axis=1)
    }

    # Return the dictionary of best performance metrics
    return best_metrics



def get_best_performance_data_tra(total_err_scores, gt_labels, topk=1, average="macro"):
    """
    Compute the best performance metrics for anomaly detection by optimizing the threshold
    and evaluating predictions against ground truth labels, specifically tailored for training-related computations.

    Args:
        total_err_scores: Error scores for the test data.
        gt_labels: Ground truth labels for the test data.
        topk: Number of top error scores to consider for each time step (default is 1).
        average: Type of averaging for metrics (default is "macro").

    Returns:
        best_metrics: A dictionary containing the best performance metrics including precision, recall, F1-score, accuracy, ROC-AUC, and threshold.
    """
    # Get the total number of features in the error scores
    total_features = total_err_scores.shape[0]

    # Identify the indices of the top-k error scores for each time step
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    # Aggregate the top-k error scores for each time step
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    # Convert ground truth labels to binary format (1 if any feature is anomalous, 0 otherwise)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Evaluate the aggregated scores and find the optimal threshold
    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, labelsFinal, 2000, return_thresold=True, average=average)
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    # Generate binary predictions based on the optimal threshold
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    # Ensure predictions and labels are integers for consistency
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        labelsFinal[i] = int(labelsFinal[i])

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labelsFinal) / len(labelsFinal)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in labelsFinal]

    # Calculate performance metrics using weighted averages
    pre = precision_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    rec = recall_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    f1 = f1_score(labelsFinal, pred_labels, sample_weight=sample_weight, average=average)
    acc = accuracy_score(labelsFinal, pred_labels, sample_weight=sample_weight)
    auc_score = roc_auc_score(labelsFinal, total_topk_err_scores, sample_weight=sample_weight, average=average)

    # Re-convert ground truth labels to binary format (redundant but ensures consistency)
    labelsFinal = (np.sum(gt_labels, axis=0) >= 1) + 0

    # Calculate average precision (AP)
    ap = average_precision_score(labelsFinal, pred_labels, average=average, sample_weight=sample_weight)

    # Compile all metrics into a dictionary
    best_metrics = {
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'accuracy': acc,
        'roc_auc': auc_score,
        'threshold': thresold,
        'AP': ap,
        'sigma_label': labelsFinal.tolist(),
        'y_pred': pred_labels.tolist(),
        'gt_labels': labelsFinal.tolist(),
        'scores_search': total_topk_err_scores.tolist(),
        'thresold': thresold,
        "scores_diff": np.mean(total_err_scores.T, axis=1)
    }

    # Return the dictionary of best performance metrics
    return best_metrics


def get_score(test_result, val_result):
    """
    Compute the best performance metrics and error scores for the test data.

    Args:
        test_result: Test result data, typically containing predictions and ground truth.
        val_result: Validation result data, used for normalization.

    Returns:
        info: A dictionary containing the best performance metrics.
        test_scores: Error scores for the test data.
    """
    # Compute full error scores and normalized scores for the test and validation data
    test_scores, normal_scores = get_full_err_scores_ori(test_result, val_result)

    # Evaluate the best performance metrics using the computed error scores
    info = get_best_performance_data_tra(test_scores, test_result[0].T, topk=1, average="macro")

    # Return the performance metrics and error scores
    return info, test_scores


def softmax(x, is_chanage=False):
    """
    Compute the Softmax function for the input array.

    Args:
        x: Input array, typically logits or unnormalized predictions.
        is_chanage: Boolean flag to determine whether to apply Softmax transformation (default is False).

    Returns:
        softmax_output: Softmax-transformed array if `is_chanage` is True; otherwise, returns the input as-is.
    """
    if is_chanage:
        # Handle numerical overflow: subtract the maximum value from the input (computed per row for batch processing)
        x_max = np.max(x, axis=-1, keepdims=True)
        
        # Compute exponential values (avoids overflow caused by high powers of e)
        exp_x = np.exp(x - x_max)
        
        # Compute the sum of exponentials for each row, to be used as the denominator
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        
        # Compute the Softmax output by normalizing the exponentials
        softmax_output = exp_x / sum_exp_x
        
        # Return the Softmax-transformed array
        return softmax_output
    else:
        # If `is_chanage` is False, return the input array unchanged
        return x



def get_val_res(scores, labels, th, path=None, search_res=None, test_data=None, val_data=None, POS="S"):
    """
    Evaluate the performance of anomaly detection and generate detailed metrics and plots.

    Args:
        scores: Anomaly scores or error scores for each time step and feature.
        labels: Ground truth labels for the data.
        th: Threshold value for determining anomalies.
        path: File path to save the results (optional).
        search_res: Additional search results for enhanced evaluation (optional).
        test_data: Test dataset for computing error scores.
        val_data: Validation dataset for computing normalized scores.
        POS: Mode indicator ("T" for training-related computations, "S" for standard).

    Returns:
        info["thresold"]: Optimal threshold determined during evaluation.
        best_metrics: Dictionary containing key performance metrics (precision, recall, F1, accuracy, ROC-AUC).
        data: Dictionary containing detailed metrics and plots for further analysis.
    """
    # Compute error scores and normalized scores based on the mode (training or standard)
    if POS == "T":
        test_scores, normal_scores = get_full_err_scores_tra(test_data, val_data)
        info = get_best_performance_data(
            total_err_scores=test_scores,
            gt_labels=test_data[0].T,
            val_scores=normal_scores,
            v_gt_labels=val_data[0].T,
            topk=1
        )
    else:
        test_scores, normal_scores = get_full_err_scores(test_data, val_data)
        info = get_best_performance_data_ori(
            total_err_scores=test_scores,
            gt_labels=test_data[0].T,
            val_scores=normal_scores,
            v_gt_labels=val_data[0].T,
            topk=1
        )

    # Initialize a dictionary to store the best performance metrics
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        "roc_auc": 0.0,
        "sigma_label": None,
    }

    # Populate the best_metrics dictionary with values from the evaluation results
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

    # Initialize a dictionary to store detailed metrics and plots
    data = {}

    # If a file path is provided and additional search results exist, generate detailed analysis
    if path is not None and "data_struct" in list(search_res.keys()):
        # Compute class weights for imbalanced datasets
        weight = np.bincount(info["gt_labels"]) / len(info["gt_labels"])
        data["gtLabels_jacc"] = info["gt_labels"]
        data["sigma_label"] = info["sigma_label"]
        data["preLabels_jacc"] = info["y_pred"].tolist()
        data["scores_jacc"] = softmax(info["scores_search"].tolist())

        # Compute precision-recall curve and average precision for Jaccard scores
        try:
            precision, recall, _ = precision_recall_curve(
                data["gtLabels_jacc"],
                softmax(data["scores_jacc"]),
                sample_weight=[weight[1] if item == 0 else weight[0] for item in data["gtLabels_jacc"]]
            )
        except:
            precision, recall, _ = precision_recall_curve(data["gtLabels_jacc"], softmax(data["scores_jacc"]))

        data["precision_jacc"] = precision[:-10].tolist()
        data["recall_jacc"] = recall[:-10].tolist()
        data["ap_jacc"] = float(average_precision_score(data["gtLabels_jacc"], softmax(data["scores_jacc"])))

        # Process structured enhancement data if available
        try:
            weight = np.bincount(search_res["data_struct"]["gtLabels_jacc_struct"]) / len(
                search_res["data_struct"]["gtLabels_jacc_struct"]
            )
            data["gtLabels_jacc_struct"] = search_res["data_struct"]["gtLabels_jacc_struct"]
            data["preLabels_jacc_struct"] = search_res["data_struct"]["preLabels_jacc_struct"].tolist()
            data["scores_jacc_struct"] = softmax(search_res["data_struct"]["scores_jacc_struct"].tolist())

            # Compute precision-recall curve and average precision for structured Jaccard scores
            try:
                precision, recall, _ = precision_recall_curve(
                    y_true=data["gtLabels_jacc_struct"],
                    probas_pred=softmax(data["scores_jacc_struct"]),
                    sample_weight=[weight[1] if item == 0 else weight[0] for item in data["gtLabels_jacc_struct"]]
                )
            except:
                precision, recall, _ = precision_recall_curve(
                    y_true=data["gtLabels_jacc_struct"],
                    probas_pred=softmax(data["scores_jacc_struct"])
                )

            data["precision_jacc_struct"] = search_res["data_struct"]["precision_jacc_struct"]
            data["recall_jacc_struct"] = search_res["data_struct"]["recall_jacc_struct"]
            data["ap_jacc_struct"] = float(
                average_precision_score(data["gtLabels_jacc_struct"], softmax(data["scores_jacc_struct"]))
            )
        except Exception as e:
            pass

        # Process search-related data
        weight = np.bincount(np.asarray(search_res["gt_labels"])) / len(np.asarray(search_res["gt_labels"]))
        data["gtLabels_search"] = search_res["gt_labels"]
        data["preLabels_search"] = search_res["y_pred"].tolist()
        data["scores_search"] = softmax(search_res["scores_search"].tolist())

        # Compute precision-recall curve and average precision for search scores
        try:
            precision, recall, _ = precision_recall_curve(
                data["gtLabels_search"],
                softmax(search_res["scores_search"]),
                sample_weight=[weight[1] if item == 0 else weight[0] for item in data["gtLabels_search"]]
            )
        except:
            precision, recall, _ = precision_recall_curve(data["gtLabels_search"], softmax(search_res["scores_search"]))

        data["precision_search"] = precision[:-10].tolist()
        data["recall_search"] = recall[:-10].tolist()
        data["ap_search"] = float(
            average_precision_score(data["gtLabels_search"], softmax(search_res["scores_search"]))
        )

        # Process POT-related data
        weight = np.bincount(search_res["lables_finals"]) / len(search_res["lables_finals"])
        data["gtLabels_pot"] = search_res["lables_finals"].tolist()
        data["preLabels_pot"] = search_res["pred_pot"].tolist()
        data["scores_pot"] = softmax(search_res["scores_pot"].tolist())

        # Compute precision-recall curve and average precision for POT scores
        try:
            precision, recall, _ = precision_recall_curve(
                data["gtLabels_pot"],
                softmax(search_res["scores_pot"]),
                sample_weight=[weight[1] if item == 0 else weight[0] for item in data["gtLabels_pot"]]
            )
        except:
            precision, recall, _ = precision_recall_curve(data["gtLabels_pot"], softmax(search_res["scores_pot"]))

        data["precision_pot"] = precision[:-10].tolist()
        data["recall_pot"] = recall[:-10].tolist()
        data["ap_pot"] = float(average_precision_score(data["gtLabels_pot"], softmax(search_res["scores_pot"])))

        # Save the detailed results to a JSON file
        p = f"{path}-pr.json"
        with open(p, mode="w") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
        fp.close()

    # If no additional search results are provided, compute basic metrics
    else:
        weight = np.bincount(info["gt_labels"]) / len(info["gt_labels"])
        data["gtLabels_jacc_struct"] = info["gt_labels"]
        data["preLabels_jacc_struct"] = info["y_pred"]
        data["scores_jacc_struct"] = info["scores_search"]
        data["sigma_label"] = info["sigma_label"]

        # Compute precision-recall curve and average precision for basic Jaccard scores
        try:
            precision, recall, _ = precision_recall_curve(
                data["gtLabels_jacc_struct"],
                softmax(data["scores_jacc_struct"]),
                sample_weight=[weight[1] if item == 0 else weight[0] for item in data["gtLabels_jacc_struct"]]
            )
        except:
            precision, recall, _ = precision_recall_curve(
                data["gtLabels_jacc_struct"],
                softmax(data["scores_jacc_struct"])
            )

        data["precision_jacc_struct"] = precision[:-10].tolist()
        data["recall_jacc_struct"] = recall[:-10].tolist()
        data["ap_jacc_struct"] = float(
            average_precision_score(data["gtLabels_jacc_struct"], softmax(data["scores_jacc_struct"]))
        )

    # Return the optimal threshold, best metrics, and detailed data
    return info["thresold"], best_metrics, data


def calculate_metrics(scores, labels, threshold):
    """
    Calculate various evaluation metrics based on a given threshold.

    Args:
        scores: Anomaly scores or error scores for each sample.
        labels: Ground truth labels for the samples.
        threshold: Threshold value to determine binary predictions.

    Returns:
        precision: Precision score computed using sklearn.
        recall: Recall score computed using sklearn.
        f1: F1-score computed using sklearn.
        accuracy: Accuracy score computed using sklearn.
        predictions: Binary predictions generated based on the threshold.
        auc_score: ROC-AUC score computed using sklearn.
    """
    # Generate binary predictions based on the threshold
    predictions = [1 if score >= threshold else 0 for score in scores]

    # Manually compute true positives, false positives, false negatives, and true negatives
    true_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    false_negatives = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    true_negatives = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    # Compute precision, recall, F1-score, and accuracy manually (commented out in favor of sklearn)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(labels)

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labels) / len(labels)
    sample_weight = None  # No weighting applied in this case

    # Compute metrics using sklearn functions with macro averaging
    precision = precision_score(labels, predictions, sample_weight=sample_weight, average="macro")
    recall = recall_score(labels, predictions, sample_weight=sample_weight, average="macro")
    f1 = f1_score(labels, predictions, sample_weight=sample_weight, average="macro")
    accuracy = accuracy_score(labels, predictions)
    auc_score = roc_auc_score(labels, scores, sample_weight=sample_weight, average="macro")

    # Return the computed metrics and predictions
    return precision, recall, f1, accuracy, predictions, auc_score

def calculateMetrics(true, pred):
    """
    Calculate the confusion matrix components (TP, FP, FN, TN) for binary classification.

    Args:
        true: Ground truth labels (binary values: 0 or 1).
        pred: Predicted labels (binary values: 0 or 1).

    Returns:
        TP: Number of true positives (correctly predicted positive cases).
        FP: Number of false positives (incorrectly predicted positive cases).
        FN: Number of false negatives (incorrectly predicted negative cases).
        TN: Number of true negatives (correctly predicted negative cases).
    """
    # Count true positives: cases where both true and predicted labels are 1
    TP = sum(1 for p, l in zip(true, pred) if p == 1 and l == 1)

    # Count false positives: cases where predicted label is 1 but true label is 0
    FP = sum(1 for p, l in zip(true, pred) if p == 1 and l == 0)

    # Count false negatives: cases where predicted label is 0 but true label is 1
    FN = sum(1 for p, l in zip(true, pred) if p == 0 and l == 1)

    # Count true negatives: cases where both true and predicted labels are 0
    TN = sum(1 for p, l in zip(true, pred) if p == 0 and l == 0)

    # Return the four components of the confusion matrix
    return TP, FP, FN, TN

def search_optimal_threshold(scores, labels, average="macro"):
    """
    Search for the optimal threshold that maximizes the F1-score and compute other metrics.

    Args:
        scores: Anomaly scores or error scores for each sample.
        labels: Ground truth labels for the samples.
        average: Type of averaging for metrics (default is "macro").

    Returns:
        best_threshold: The threshold that maximizes the F1-score.
        best_metrics: A dictionary containing the best performance metrics including precision, recall, F1-score, accuracy, ROC-AUC, and the optimal threshold.
    """
    # Rank the scores ordinally to facilitate threshold-based predictions
    scores_sorted = rankdata(scores, method='ordinal')

    # Initialize variables to track the best F1-score and corresponding threshold
    best_f1 = 0
    best_threshold = 0

    # Initialize a dictionary to store the best metrics
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'roc_auc': 0.0
    }

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labels) / len(labels)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in labels]

    # Define the search range for the threshold
    low, high = min(scores), max(scores)
    eps = 1e-8  # Small epsilon to control the precision of the search

    # Perform binary search to find the optimal threshold
    while high - low > eps:
        mid = (low + high) / 2  # Midpoint of the current search range
        predictions = (scores_sorted > mid).astype(int)  # Generate binary predictions based on the midpoint

        # Compute precision, recall, and F1-score for the current threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, sample_weight=sample_weight
        )

        # Update the best metrics if the current F1-score is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = mid
            best_metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy_score(labels, predictions),
                'roc_auc': roc_auc_score(labels, scores, sample_weight=sample_weight),
                'predictions': predictions,
                'threshold': best_threshold
            })

        # Adjust the search range based on the F1-score comparison
        low, high = (mid, high) if f1 > best_f1 else (low, mid)

    # Return the optimal threshold and the best metrics
    return best_threshold, best_metrics