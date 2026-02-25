# Replicated from the following paper:
# Nakamura, T., Imamura, M., Mercer, R. and Keogh, E., 2020, November. 
# MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive 
# Time Series Archives. In 2020 IEEE International Conference on Data Mining (ICDM) 
# (pp. 1190-1195). IEEE.
import json

import numpy as np
from pprint import pprint
from time import time
from src.utils import *
from src.constants import *
from src.diagnosis import *
from src.pot import *
maxint = 200000

# z-normalized euclidean distance
def dist(t, q):
	m = q.shape[0]
	# t, q = t.reshape(-1), q.reshape(-1)
	# znorm2 = 2 * m * (1 - (np.dot(q, t) - m * np.mean(q) * np.mean(t)) / (m * np.std(q) * np.std(t)))
	znorm2 = np.mean((t - q) ** 2)
	return np.sqrt(znorm2)

# get L length subsequence from t starting at i
def getsub(t, L, i):
	return t[i:i+L]

# Candidate Selection Algorithm
def csa(t, L, r):
	C = []
	for i in range(1, t.shape[0] - L + 1):
		iscandidate = True
		for j in C:
			if i != j:
				if dist(getsub(t, L, i), getsub(t, L, j)) < r:
					C.remove(j)
					iscandidate = False
		if iscandidate and i not in C:
			C.append(i)
	if C:
		return C
	else:
		return []

# Checking function
def check(t, pred):
	labels = [];
	Scores = None;
	for i in range(t.shape[1]):
		new = np.convolve(t[:, i], np.ones(cvp)/cvp, mode='same')
		scores = np.abs(new - t[:,i])
		if i == 0 :
			Scores = scores
		else:
			Scores += scores
		labels.append((scores > np.percentile(scores, percentile_merlin)) + 0)
	labels = np.array(labels).transpose()
	return (np.sum(labels, axis=1) >= 1) + 0, labels, Scores / t.shape[1]

# Discords Refinement Algorithm
def drag(C, t, L, r):
	D = [];
	if not C: return []
	for i in range(1, t.shape[0] - L + 1):
		isdiscord = True
		dj = maxint
		for j in C:
			if i != j:
				d = dist(getsub(t, L, i), getsub(t, L, j))
				if d < r:
					C.remove(j)
					isdiscord = False
				else:
					dj = min(dj, d)
		if isdiscord:
			D.append((i, L, dj))
	return D

# MERLIN
def merlin(t, minL, maxL):
	r = 2 * np.sqrt(minL)
	dminL = - maxint; DFinal = []
	while dminL < 0:
		C = csa(t, minL, r)
		D = drag(C, t, minL, r)
		r = r / 2
		if D: break
	rstart = r
	distances = [-maxint] * 4
	
	for i in range(minL, min(minL+4, maxL)):
		di = distances[i - minL]
		dim1 = rstart if i == minL else distances[i - minL - 1]
		r = 0.99 * dim1
		while di < 0:
			C = csa(t, i, r)
			D = drag(C, t, i, r)
			if D:
				di = np.max([p[2] for p in D])
				distances[i - minL] = di
				DFinal += D
			r = r * 0.99
		
	
	for i in range(minL + 4, maxL + 1):
		M = np.mean(distances)
		S = np.std(distances) + 1e-2
		r = M - 2 * S
		di = - maxint
		for _ in range(1000):
			C = csa(t, i, r)
			D = drag(C, t, i, r)
			if D:
				di = np.max([p[2] for p in D])
				DFinal += D
				if di > 0:	break
			r = r - S
	vals = []
	for p in DFinal:
		if p[2] != maxint: vals.append(p[2])
	dmin = np.argmax(vals)
	return DFinal[dmin], DFinal

def get_result(pred, labels):
	p_t = calc_point2point(pred, labels)
	result = {
        'f1': float(p_t[0]),
        'Precision': float(p_t[1]),
        'Recall': float(p_t[2]),
        'TP': float(p_t[3]),
        'TN': float(p_t[4]),
        'FP': float(p_t[5]),
        'FN': float(p_t[6]),
        'ROC/AUC': float(p_t[7]),
		'ACC':float(p_t[8]),
    }
	return result



def run_merlin(test, labels, dset, folder):
    """
    Run the MERLIN algorithm for anomaly detection and save the results.

    Args:
        test: Test dataset (PyTorch DataLoader or similar iterable).
        labels: Ground truth labels for anomalies.
        dset: Dataset name (used for conditional metric updates).
        folder: Directory path to save the results.

    Returns:
        None: Saves results to a JSON file.
    """
    # Extract the first batch of test data and convert to NumPy array
    t = next(iter(test)).detach().numpy()
    labelsAll = labels  # Store full labels for later use

    # Convert multi-dimensional labels to binary (1 if any feature is anomalous, 0 otherwise)
    labels = (np.sum(labels, axis=1) >= 1) + 0
    lsum = np.sum(labels)  # Total number of anomalies

    # Record start time for performance measurement
    start = time()

    # Initialize prediction array
    pred = np.zeros_like(labels)

    # Run MERLIN algorithm to detect anomalies
    d, _ = merlin(t, 60, 62)  # Detect anomalies with subsequence length range [60, 62]
    pred[d[0]:d[0] + d[1]] = 1  # Mark detected anomaly region

    # Refine predictions and compute anomaly scores
    pred, predAll, scores = check(t, pred)

    # Compute evaluation metrics
    result = get_result(pred, labels)

    # Update results with Hit@p% and NDCG@p% metrics if applicable
    result.update(hit_att(predAll, labelsAll))
    result.update(ndcg(predAll, labelsAll))

    # Compute class weights for imbalanced datasets
    weight = np.bincount(labels) / len(labels)

    # Add ground truth labels, predictions, and scores to results
    result["gtLabels"] = labels.tolist()
    result["preLabels"] = pred.tolist()
    result["scores"] = scores.tolist()

    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(
        labels, scores, sample_weight=[weight[1] if item == 0 else weight[0] for item in labels]
    )
    result["precision"] = precision[:-10].tolist()
    result["recall"] = recall[:-10].tolist()
    result["ap_pre"] = float(average_precision_score(labels, scores))

    # Compute average training time per sample
    result["train_time"] = float((time() - start) / len(labels))

    # Save results to a JSON file
    try:
        os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
        p = f"{folder}/result.json"
        with open(p, mode="w") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=4)
        fp.close()
        print("Merlin results saved successfully, save path is: " ,p)
    except Exception as e:
        print("Error saving Merlin results:", e)
	
