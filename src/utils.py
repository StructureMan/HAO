import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd
import numpy as np


class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def getresults2(df, result):
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall']:
        results2[a] = df2[a]
    results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2
