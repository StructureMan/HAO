import numpy as np
from sklearn.metrics import ndcg_score
from src.constants import lm

def hit_att(ascore, labels, ps=[500, 600]):
    """
    Calculate the Hit@p% metric for anomaly detection performance.

    Args:
        ascore: Array of anomaly scores for each sample.
        labels: Ground truth labels indicating anomalies (1 for anomaly, 0 for normal).
        ps: List of percentages (e.g., 500, 600) to evaluate Hit@p% (default is [500, 600]).

    Returns:
        res: Dictionary containing Hit@p% scores for each specified percentage.
    """
    res = {}  # Initialize a dictionary to store results for each percentage

    # Iterate over each specified percentage
    for p in ps:
        hit_score = []  # List to store Hit scores for the current percentage

        # Iterate over each sample
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]  # Extract anomaly scores and labels for the current sample

            # Sort anomaly scores in descending order and identify indices of actual anomalies
            a = np.argsort(a).tolist()[::-1]  # Indices of scores sorted in descending order
            l = set(np.where(l == 1)[0])      # Set of indices where labels indicate anomalies

            # Proceed only if there are actual anomalies in the sample
            if l:
                # Determine the size of the top-p% predictions
                size = round(p * len(l) / 100)
                a_p = set(a[:size])  # Top-p% predicted anomaly indices

                # Calculate the intersection between predicted and actual anomalies
                intersect = a_p.intersection(l)

                # Compute the Hit ratio: proportion of actual anomalies correctly identified
                hit = len(intersect) / len(l)
                hit_score.append(hit)  # Store the Hit ratio for this sample

        # Compute the mean Hit@p% across all samples and store in the result dictionary
        res[f'Hit@{p}%'] = np.mean(hit_score)

    return res  # Return the dictionary of Hit@p% scores

def ndcg(ascore, labels, ps=[500, 600]):
    """
    Calculate the NDCG@p% metric for anomaly detection performance.

    Args:
        ascore: Array of anomaly scores for each sample.
        labels: Ground truth labels indicating anomalies (1 for anomaly, 0 for normal).
        ps: List of percentages (e.g., 500, 600) to evaluate NDCG@p% (default is [500, 600]).

    Returns:
        res: Dictionary containing NDCG@p% scores for each specified percentage.
    """
    res = {}  # Initialize a dictionary to store results for each percentage

    # Iterate over each specified percentage
    for p in ps:
        ndcg_scores = []  # List to store NDCG scores for the current percentage

        # Iterate over each sample
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]  # Extract anomaly scores and labels for the current sample

            # Identify indices of actual anomalies
            labs = list(np.where(l == 1)[0])

            # Proceed only if there are actual anomalies in the sample
            if labs:
                # Determine the cutoff rank (k_p) for NDCG calculation
                k_p = round(p * len(labs) / 100)

                try:
                    # Compute NDCG score using sklearn's ndcg_score function
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
                except Exception as e:
                    # Print error and return an empty dictionary if NDCG calculation fails
                    print(e)
                    return {}

                # Store the NDCG score for this sample
                ndcg_scores.append(hit)

        # Compute the mean NDCG@p% across all samples and store in the result dictionary
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)

    return res  # Return the dictionary of NDCG@p% scores



