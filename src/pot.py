import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)

    weight = np.bincount(actual) / len(actual)
    sample_weight = [weight[1] if item == 0 else weight[0] for item in actual]

    precision = precision_score(actual, predict, sample_weight=sample_weight, average="macro")
    recall = recall_score(actual, predict, sample_weight=sample_weight, average="macro")
    f1 = f1_score(actual, predict, sample_weight=sample_weight, average="macro")
    acc = accuracy_score(actual, predict,sample_weight=sample_weight)
    roc_auc = roc_auc_score(actual, predict, sample_weight=sample_weight, average="macro")
    ap = average_precision_score(actual, predict, average="macro", sample_weight=sample_weight)
    return f1, precision, recall, TP, TN, FP, FN, roc_auc,acc,ap


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = (score > threshold) + 0
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    # return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=0.001, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    lms = lm[0]
    max_step = 0
    while True:
        max_step += 1
        if max_step > 20:
            break
        try:
            # print(lms)

            s = SPOT(q)  # SPOT object

            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step



        except Exception as e:
            # print(e)
            lms = lms * 0.999

        else: break
    try:
        ret = s.run(dynamic=True)  # run
        # print(len(ret['alarms']))
        # print(len(ret['thresholds']))
        pot_th = np.mean(ret['thresholds']) * lm[1]
        # pot_th = np.percentile(score, 100 * lm[0])
        # np.percentile(score, 100 * lm[0])
        pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
        # DEBUG - np.save(f'{debug}.npy', np.array(pred))
        # DEBUG - print(np.argwhere(np.array(pred)))
        p_t = calc_point2point(pred, label)

        return {
            'f1': p_t[0],
            'precision': p_t[1],
            'recall': p_t[2],
            'TP': p_t[3],
            'TN': p_t[4],
            'FP': p_t[5],
            'FN': p_t[6],
            'p_latency': p_latency,
            'ROC/AUC': p_t[7],
            'threshold': pot_th,
            'ACC': p_t[8],
            'AP': p_t[9],
            "lables_finals":label,
            "scores_pot":score,
            "pred_pot":pred
            # 'pot-latency': p_latency
        }, np.array(pred)
    except:
        try:
            pot_th = 0.0
            # pot_th = np.percentile(score, 100 * lm[0])
            # np.percentile(score, 100 * lm[0])
            pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
            # print("p_latency", p_latency)
            # DEBUG - np.save(f'{debug}.npy', np.array(pred))
            # DEBUG - print(np.argwhere(np.array(pred)))
            p_t = calc_point2point(pred, label)
            return {
                'f1': p_t[0],
                'precision': p_t[1],
                'recall': p_t[2],
                'TP': p_t[3],
                'TN': p_t[4],
                'FP': p_t[5],
                'FN': p_t[6],
                'p_latency': p_latency,
                'ROC/AUC': p_t[7],
                'threshold': pot_th,
                'ACC': p_t[8],
                'AP': p_t[9],
                "lables_finals": label,
                "scores_pot": score,
                "pred_pot": pred
            }, np.array(pred)
        except:
            return {
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'TP': 0.0,
                'TN': 0.0,
                'FP': 0.0,
                'FN': 0.0,
                'ROC/AUC': 0.0,
                'threshold': 0.0,
                'p_latency': 0.0,
                'ACC': 0.0,
                'AP': 0.0,
            }, np.zeros(shape=len(label))