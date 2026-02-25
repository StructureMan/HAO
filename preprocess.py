import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile
import matplotlib.pyplot as plt
import re
datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', "ASD"]
import wfdb
from scipy.ndimage import gaussian_filter1d
# 统计并处理低正例比例的列
# def plotdata(train,test,label,dataset,sub_dataset):
#     plt.clf()
#     num_subfig = 1
#     print(num_subfig)
#     num_subfig = num_subfig * 3
#     plt.figure(figsize=(30, 3 * num_subfig))
#     op_label = np.zeros(shape=label.shape)
    
#     for item_fig in range(1, num_subfig , 3):
#         # item_fig = 1
#         data_item = item_fig // 3
#         plt.subplot(num_subfig, 1, item_fig)
#         plt.plot(train.T[data_item], c="blue", alpha=0.6, label=f"Train Data \n Sensor:{data_item}", linewidth=1.5)
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#         plt.subplot(num_subfig, 1, item_fig + 1)
#         plt.plot(test.T[data_item], c="black", alpha=0.6, label="Test Data", linewidth=1.5)
#         plt.fill_between(np.arange(label.shape[0]), label.T[data_item ], color='red', alpha=0.3,
#                          label=f"GT Sensor:{data_item}")
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#         plt.subplot(num_subfig, 1, item_fig + 2)
#         plt.plot(test.T[data_item ], c="black", alpha=0.6, label="Test Data", linewidth=1.5)
#         plt.fill_between(np.arange(op_label.shape[0]), op_label.T[data_item ], color='blue', alpha=0.3,
#                          label=f"OT Sensor:{data_item}")
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         break
#     # plt.fill_between(np.arange(labels.shape[0]), labels_val,
#     #                  color='yellow', alpha=0.2, label="Val truth")
#     # plt.fill_between(np.arange(labels.shape[0]), labels_val_pre,
#     #                  color='blue', alpha=0.1, label="Val Pre")
#     # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         # break
#     # plt.subplot(num_subfig, 1, true.shape[1] + 1)
#     # plt.plot(Struct_mess["Anomaly_max"], c="green", alpha=0.6, label="Rec Anomaly", linewidth=1.5)
#     # plt.fill_between(np.arange(labels.shape[0]), labels.T[0] * Struct_mess["Anomaly_max"],
#     #                  color='red', alpha=0.3, label="Ground truth")
#     # # plt.plot(np.asarray(Struct_mess["scores_search"]).T, c="blue", alpha=0.6, label="SAnomaly Score", linewidth=1.5)
#     # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


#     path = '{}-{}.png'.format(dataset,sub_dataset)
#     print(path)
#     plt.savefig(path)
from tqdm import tqdm
def validate_test_with_sliding_window_advanced(train, test, window_size=100, threshold=0.8, method='distance'):
    """
    优化版本的滑动窗口验证（完整数据比较）
    """
    n_test, n_features = test.shape
    n_train = train.shape[0]
    op_label = np.zeros((n_test, n_features))
    # tqdm(enumerate(data), total=len(data), desc=f'Epoch {epoch}'):
    for feature_idx in range(n_features):
        train_feature = train[:, feature_idx]
        test_feature = test[:, feature_idx]

        # 计算训练数据的基本统计信息
        train_mean = np.mean(train_feature)
        train_std = np.std(train_feature)

        # 向量化处理：预先计算所有测试窗口
        test_windows = np.zeros((n_test, window_size))
        for  i in range(n_test):
            if i >= window_size:
                test_windows[i] = test_feature[i - window_size:i]
            else:
                test_windows[i] = np.concatenate([
                    np.repeat(test_feature[0:1], window_size - i, axis=0),
                    test_feature[0:i]
                ], axis=0)

        # 预先计算测试窗口的统计信息
        test_window_means = np.mean(test_windows, axis=1)
        test_window_stds = np.std(test_windows, axis=1)

        for index, i in tqdm(enumerate(range(n_test)), total=n_test,desc=f"Feature:{feature_idx}"):
            window_data = test_windows[i]
            window_mean = test_window_means[i]
            window_std = test_window_stds[i]

            max_metric = -1 if method == 'correlation' or method == 'distribution' else float('inf')

            # 对所有训练样本进行比较
            for j in range(n_train):
                if j >= window_size:
                    train_data = train_feature[j - window_size:j]
                else:
                    train_data = np.concatenate([
                        np.repeat(train_feature[0:1], window_size - j, axis=0),
                        train_feature[0:j]
                    ], axis=0)

                if method == 'correlation':
                    try:
                        if np.std(train_data) > 1e-8 and window_std > 1e-8:
                            correlation = np.corrcoef(train_data, window_data)[0, 1]
                            if not np.isnan(correlation):
                                metric = abs(correlation)
                                max_metric = max(max_metric, metric)
                    except:
                        pass

                elif method == 'distance':
                    try:
                        distance = np.linalg.norm(train_data - window_data)
                        metric = np.tanh(distance)
                        if max_metric == float('inf'):
                            max_metric = metric
                        else:
                            max_metric = min(max_metric, metric)  # 距离越小越好
                    except:
                        pass

                elif method == 'distribution':
                    try:
                        train_window_mean = np.mean(train_data)
                        train_window_std = np.std(train_data)

                        if train_std > 1e-8 and train_window_std > 1e-8:
                            mean_diff = abs(window_mean - train_window_mean) / (train_std + 1e-8)
                            std_diff = abs(train_window_std - train_window_std) / (train_std + 1e-8)
                            distribution_diff = (mean_diff + std_diff) / 2
                            metric = 1 / (1 + distribution_diff)
                            max_metric = max(max_metric, metric)
                    except:
                        pass

            # 判断是否异常
            if method == 'correlation' and max_metric < threshold:
                op_label[i, feature_idx] = 1
            elif method == 'distance' and max_metric > threshold:
                op_label[i, feature_idx] = 1
            elif method == 'distribution' and max_metric < threshold:
                op_label[i, feature_idx] = 1

    return op_label

def validate_test_with_sliding_window(train, test, window_size=100, threshold=0.8):
    """
    基于滑动窗口验证测试数据与训练数据的相关性
    
    参数:
        train: 训练数据，形状为 [n_train, n_features]
        test: 测试数据，形状为 [n_test, n_features]
        window_size: 滑动窗口大小
        threshold: 相关性阈值，低于此值则标记为异常
    
    返回:
        op_label: 异常标签，形状为 [n_test, n_features]
    """
    n_test, n_features = test.shape
    op_label = np.zeros((n_test, n_features))
    
    # 对每个特征分别处理
    for feature_idx in range(n_features):
        train_feature = train[:, feature_idx]
        test_feature = test[:, feature_idx]
        
        # 对测试数据使用滑动窗口
        for i in range(n_test):
            # 确定窗口范围
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n_test, i + window_size // 2)
            
            # 获取当前窗口的数据
            window_data = test_feature[start_idx:end_idx]
            
            # 计算窗口数据与训练数据的相关性
            # 使用皮尔逊相关系数
            if len(window_data) > 1 and np.std(train_feature) > 1e-8 and np.std(window_data) > 1e-8:
                # 确保两个数组长度一致
                compare_length = min(len(train_feature), len(window_data))
                train_sample = train_feature[:compare_length]
                window_sample = window_data[:compare_length]
                
                try:
                    correlation = np.corrcoef(train_sample, window_sample)[0, 1]
                    # 如果相关性为NaN，设为0
                    if np.isnan(correlation):
                        correlation = 0
                except Exception as e:
                    print(f"计算相关系数时出错: {e}")
                    correlation = 0
            else:
                correlation = 0
            
            # 如果相关性低于阈值，则标记为异常
            if abs(correlation) < threshold:
                op_label[i, feature_idx] = 1
    
    return op_label

def sliding_window_validation(train, test, labels, window_size=100, threshold=0.8, method='distance'):
    """
    滑动窗口验证主函数
    
    参数:
        train: 训练数据
        test: 测试数据
        labels: 真实标签
        window_size: 滑动窗口大小
        threshold: 阈值
        method: 验证方法
    
    返回:
        op_label: 预测的异常标签
    """
    print(f"开始滑动窗口验证: 窗口大小={window_size}, 阈值={threshold}, 方法={method}")
    
    # 执行验证
    op_label = validate_test_with_sliding_window_advanced(
        train, test, window_size=window_size, threshold=threshold, method=method
    )
    
    # 统计结果
    total_points = op_label.shape[0] * op_label.shape[1]
    anomaly_points = np.sum(op_label)
    
    print(f"验证完成:")
    print(f"  - 总数据点数: {total_points}")
    print(f"  - 检测到异常点数: {anomaly_points}")
    print(f"  - 异常比例: {anomaly_points/total_points:.4f}")
    
    # 如果有真实标签，计算评估指标
    if labels is not None:
        # 确保标签维度一致
        if labels.shape != op_label.shape:
            print(f"标签维度不匹配: 真实标签{labels.shape} vs 预测标签{op_label.shape}")
        else:
            # 计算基本的评估指标
            tp = np.sum((op_label == 1) & (labels == 1))
            fp = np.sum((op_label == 1) & (labels == 0))
            tn = np.sum((op_label == 0) & (labels == 0))
            fn = np.sum((op_label == 0) & (labels == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            
            print(f"评估指标:")
            print(f"  - 精确率 (Precision): {precision:.4f}")
            print(f"  - 召回率 (Recall): {recall:.4f}")
            print(f"  - F1分数: {f1_score:.4f}")
    
    return op_label
# 在您的 plotdata 函数中使用这个验证功能
def plotdata_with_validation(train, test, label, dataset, sub_dataset, window_size=100, threshold=0.99,path="",desc=""):
    """
    绘制数据图并进行滑动窗口验证
    """
    # 执行滑动窗口验证
    # op_label = sliding_window_validation(train, test, label, window_size, threshold)
    
    # 绘制结果（保持原有绘图逻辑）
    plt.clf()
    num_subfig = train.shape[1]
    # print(num_subfig)
    num_subfig = num_subfig * 3
    plt.figure(figsize=(30, 1 * num_subfig),dpi=100)
    op_label = label
    downsample_rate = 1
    for item_fig in range(1, num_subfig, 3):
        data_item = item_fig // 3
        # if data_item >= train.shape[1]:  # 避免超出特征维度
        #     break
            
        plt.subplot(num_subfig, 1, item_fig)
        plt.plot(train.T[data_item][::downsample_rate], c="blue", alpha=0.6, label=f"Train Data \n Sensor:{data_item + 1 }", linewidth=1.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.subplot(num_subfig, 1, item_fig + 1)
        plt.plot(test.T[data_item][::downsample_rate], c="black", alpha=0.6, label="Test Data", linewidth=1.5)
        plt.fill_between(np.arange(label.shape[0]), label.T[data_item], color='red', alpha=0.3,
                         label=f"GT Sensor:{data_item + 1}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.subplot(num_subfig, 1, item_fig + 2)
        plt.plot(test.T[data_item][::downsample_rate], c="black", alpha=0.6, label="Test Data", linewidth=1.5)
        plt.fill_between(np.arange(op_label.shape[0]), op_label.T[data_item], color='blue', alpha=0.3,
                         label=f"OT Sensor:{data_item + 1}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # break
    plt.tight_layout()
    path = '{}/{}-{}-{}-validation-{}-{}.png'.format(path,dataset, sub_dataset, window_size,threshold,desc)
    print(path)
    plt.savefig(path)
    plt.close()
    return op_label


def filter_columns_by_positive_rate(train_data, test_data, threshold=0.5):
    """
    根据正例率过滤列
    """
    print("处理前各列统计信息:")
    for i in range(train_data.shape[1]):
        train_pos_rate = np.mean(train_data[:, i] == 1) if len(np.unique(train_data[:, i])) <= 3 else -1
        test_pos_rate = np.mean(test_data[:, i] == 1) if len(np.unique(test_data[:, i])) <= 3 else -1
        if train_pos_rate >= 0:  # 是二值数据
            print(f"列 {i}: 训练集正例率={train_pos_rate:.3f}, 测试集正例率={test_pos_rate:.3f}")
    
    # 计算训练集中每列的正例率
    positive_rates = np.mean(train_data == 1, axis=0)
    
    # 标记正例率低于阈值的列
    low_positive_columns = positive_rates < threshold
    
    # 统计信息
    total_columns = train_data.shape[1]
    filtered_columns = np.sum(low_positive_columns)
    print(f"\n总列数: {total_columns}")
    print(f"正例率低于{threshold}的列数: {filtered_columns}")
    print(f"过滤比例: {filtered_columns/total_columns:.2%}")
    
    # 将低正例率的列置为0
    train_data[:, low_positive_columns] = 0
    test_data[:, low_positive_columns] = 0
    
    return train_data, test_data


def process_binary_columns(train_data, test_data, positive_threshold=0.5):
    """
    处理可能的二值数据列
    """
    for col_idx in range(train_data.shape[1]):
        # 检查该列是否为二值数据（主要包含0和1）
        train_unique = np.unique(train_data[:, col_idx])
        test_unique = np.unique(test_data[:, col_idx])

        # 判断是否为二值特征
        is_binary = (len(train_unique) <= 3 and
                     set(train_unique).issubset({0, 1, np.nan}) and
                     len(test_unique) <= 3 and
                     set(test_unique).issubset({0, 1, np.nan}))

        if is_binary:
            # 计算正例（值为1）的比例
            train_positive_count = np.sum(train_data[:, col_idx] == 1)
            train_total_count = len(train_data[:, col_idx])
            train_positive_ratio = train_positive_count / train_total_count if train_total_count > 0 else 0

            # 如果正例比例低于阈值，将该列置为0
            if train_positive_ratio < positive_threshold:
                train_data[:, col_idx] = 0
                test_data[:, col_idx] = 0
                print(f"列 {col_idx} 正例比例 {train_positive_ratio:.3f} < {positive_threshold}，已置为0")

    return train_data, test_data
wadi_drop = ['1_LS_001_AL',
             '1_LS_002_AL',
             '1_MV_001_STATUS',
             '1_MV_002_STATUS',
             '1_MV_003_STATUS',
             '1_MV_004_STATUS',
             '1_MV_004_STATUS',
             '1_P_001_STATUS',
             '1_P_002_STATUS',
             '1_P_002_STATUS',
             '1_P_003_STATUS',
             '1_P_004_STATUS',
             '1_P_005_STATUS',
             '1_P_006_STATUS',
             '2_LS_001_AL',
             '2_LS_002_AL',
             '2_LS_101_AH',
             '2_LS_101_AL',
             '2_LS_201_AH',
             '2_LS_201_AL',
             '2_LS_301_AH',
             '2_LS_301_AL',
             '2_LS_401_AH',
             '2_LS_401_AL',
             '2_LS_501_AH',
             '2_LS_501_AL',
             '2_LS_601_AH',
             '2_LS_601_AL',
             '2_MCV_007_CO',
             '2_MCV_101_CO',
             '2_MCV_201_CO',
             '2_MCV_401_CO',
             '2_MCV_401_CO',
             '2_MV_001_STATUS',
             '2_MV_002_STATUS',
             '2_MV_003_STATUS',
             '2_MV_004_STATUS',
             '2_MV_005_STATUS',
             '2_MV_006_STATUS',
             '2_MV_009_STATUS',
             '2_MV_101_STATUS',
             '2_MV_201_STATUS',
             '2_MV_301_STATUS',
             '2_MV_401_STATUS',
             '2_MV_501_STATUS',
             '2_MV_601_STATUS',
             '2_MV_601_STATUS',
             '2_P_001_STATUS',
             '2_P_002_STATUS',
             '2_P_003_STATUS',
             '2_P_004_STATUS',
             '2_SV_101_STATUS',
             '2_SV_201_STATUS',
             '2_SV_301_STATUS',
             '2_SV_401_STATUS',
             '2_SV_501_STATUS',
             '3_AIT_001_PV',
             '3_LS_001_AL',
             '3_MV_001_STATUS',
             '3_MV_002_STATUS',
             '3_MV_003_STATUS',
             '3_P_001_STATUS',
             '3_P_002_STATUS',
             '3_P_003_STATUS',
             '3_P_004_STATUS',
             'PLANT_START_STOP_LOG',
             'PLANT_START_STOP_LOG',
             ]

def read_arff(file_path):
    with open(file_path, 'r') as file:
        data = []
        header = []
        relation = ''
        for line in file:
            line = line.strip()
            if line.startswith('@relation'):
                relation = line.split(maxsplit=1)[1]
            elif line.startswith('@attribute'):
                attribute_name = line.split(maxsplit=1)[1].split(maxsplit=1)[0]
                attribute_type = line.split(maxsplit=1)[1].split(maxsplit=1)[1]
                header.append((attribute_name, attribute_type))
            elif line.startswith('@data'):
                continue
            elif not line.startswith('%') and line:
                # Remove spaces and newlines, then split by commas
                instance = [value.strip() for value in line.split(',')]
                data.append(instance)
        return relation, header, data

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape, temp


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
        temp[start - 1:end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    np.savetxt(os.path.join(output_folder, f"SMD/{dataset}_{category}.csv"), temp, delimiter=",")
    # np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    return temp


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def standnormalize(a, average=None, stderr=None):
    if average is None: average, stderr = np.average(a, axis=0), np.std(a, axis=0)
    return (a - average) / (stderr + 0.0001), average, stderr


def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        # train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)

        # plt.plot(labels)
        #
        # plt.title('Binary Image')  # 设置标题
        # plt.savefig("WADI.png")
        print(len(np.where(np.sum(labels, axis=1) > 0)[0]))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        total_train = []
        total_test = []
        total_a = []
        for filename in file_list:
            data_tr,data_te,l = [],[],[]
            if filename.endswith('.txt'):
                tr,data_tr = load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                total_train.append(tr[0])
                s,data_te = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                total_test.append(s[0])
                l = load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
                total_a.append(len(np.where(np.sum(np.asarray(l), axis=1) > 0)[0]))
                print("neg_num:", len(np.where(np.sum(np.asarray(l), axis=1) > 0)[0]))
            print(filename,"**********************")
            if filename == "machine-1-1.txt":
                plotdata_with_validation(data_tr, data_te, l, dataset, filename, window_size=5, threshold=0.0001,
                                         path=folder, desc="check_data")
            print(filename)
        print(dataset, "train-len:", sum(total_train))
        print(dataset, "test-len:", sum(total_test))
        print(dataset, "ar:", sum(total_a) / sum(total_test))
    elif dataset == 'GECCO':
        dataset_folder = 'data/GECCO/'
        print(dataset_folder)

        # 使用 rpy2 读取 RDS 文件

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import rpy2py
        # 激活 pandas2ri 转换器
        pandas2ri.activate()

        # 读取 RDS 文件
        # readRDS = robjects.r['readRDS']
        rds_file_path = os.path.join(dataset_folder, 'waterDataTraining.RDS')
        data = robjects.r['readRDS'](rds_file_path)
        data = data.rx(True, data.colnames[1:])
        # 将R数据框转换为pandas数据框
        pandas_df = rpy2py(data)
        data = pandas_df.values
        use_data = []

        for item in data:
            # print(item)
            if not item[-1] :
                use_data.append(item[:-1])
        train = np.asarray(use_data).astype(np.float64)
        train = np.nan_to_num(train)
        train, _,_ = normalize3(train)

        rds_file_path = os.path.join(dataset_folder, 'waterDataTestingUpload.RDS')
        data = robjects.r['readRDS'](rds_file_path)
        data = data.rx(True, data.colnames[1:])
        # 将R数据框转换为pandas数据框
        pandas_df = rpy2py(data)
        data = pandas_df.values

        test = np.asarray(data[:,:-1]).astype(np.float64)
        test = np.nan_to_num(test)
        test, _, _ = normalize3(test)
        labels = np.expand_dims(data[:,-1],axis=1)

        print(labels.shape, test.shape, train.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
        #
        # print(f"数据已保存到 {csv_file}")
    elif dataset == 'LANL':
        dataset_folder = 'data/LANL/'
        print(dataset_folder)
        # 使用 pandas 分块读取文件
        chunk_size = 100000  # 根据实际情况调整块大小
        flows_data = pd.read_csv(os.path.join(dataset_folder, "flows.txt"), header=None, chunksize=chunk_size)
        # redteam_data = pd.read_csv(os.path.join(dataset_folder, "redteam.txt"), header=None, chunksize=chunk_size)

        redteam_data = np.load(os.path.join(dataset_folder, "redteam.npy"),allow_pickle=True)
        time_redteam = redteam_data[:,0].astype(np.float64)
        print(min(time_redteam),max(time_redteam),len(time_redteam))
        print(redteam_data)

        # 获取正常数据
        # 处理 flows.txt
        flows_list = []
        for chunk in flows_data:
            # 确保每一列都是字符串类型
            chunk = chunk.apply(lambda x: x.astype(str).str.replace(r'[\n ]', '', regex=True))
            flows_list.append(chunk)
            print(flows_list)
            break
        flows_df = pd.concat(flows_list)
        flows_df.to_numpy()
        np.save(os.path.join(dataset_folder, "flows.npy"), flows_df.to_numpy(), allow_pickle=True)
        data = flows_df.to_numpy()
        servers  = set(data[:,2])
        print(servers)
        print(len(servers))
        # 处理 redteam.txt
        # redteam_list = []
        # for chunk in redteam_data:
        #     # 确保每一列都是字符串类型
        #     chunk = chunk.apply(lambda x: x.astype(str).str.replace(r'[\n ]', '', regex=True))
        #     redteam_list.append(chunk)
        # redteam_df = pd.concat(redteam_list)
        # redteam_df.to_numpy()
        # np.save(os.path.join(dataset_folder, "redteam.npy"), redteam_df.to_numpy(), allow_pickle=True)



    elif dataset == 'WH':
        dataset_folder = 'data/WH/'
        train = pd.read_csv(os.path.join(dataset_folder, 'Train.csv'))

        train = train.values
        test_ori_data = train
        use_data = []
        for item in train[:train.shape[0] // 2, :]:
            if item[-1] == 0:
                use_data.append(item[:-1])
        train = np.asarray(use_data).astype(np.float64)
        train, min_a, max_a = normalize3(train)

        test = test_ori_data[test_ori_data.shape[0] // 2:, :-1]
        # _, min_a, max_a = normalize3(np.concatenate((train, test), axis=0))
        # train, _, _ = normalize3(train,min_a,max_a)
        test,_,_ = normalize3(test,min_a,max_a)
        labels = np.expand_dims(test_ori_data[test_ori_data.shape[0] // 2:, -1],axis=1)


        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        print(labels)
        print(labels.shape, test.shape, train.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'NSL':
        dataset_folder = 'data/NSL/'

        # 加载测试数据和训练数据
        train = pd.read_csv(dataset_folder + "KDDTrain+.txt", sep=',', header=None)  # 假设数据以逗号分隔
        print(train.describe())
        print(train)
        # 离散值给值
        train_data = train.values

        # 协议类型编码
        protocol_type = list(set(train_data[:,1]))
        protocol_type_code = {}
        for i,item in enumerate(protocol_type):
            protocol_type_code[item] = i
        print(protocol_type_code)

        # service code
        service = list(set(train_data[:, 2]))
        service_code = {}
        for i, item in enumerate(service):
            service_code[item] = i
        print(service_code)

        # flag code
        flag = list(set(train_data[:, 3]))
        flag_code = {}
        for i, item in enumerate(flag):
            flag_code[item] = i
        print(flag_code)

        # normal code
        normal = list(set(train_data[:, 41]))
        normal_code = {}
        for i, item in enumerate(normal):
            normal_code[item] = i
        print(normal_code)
        print(train_data.shape)
        used_data = []
        for item in train_data:
            item[1] = protocol_type_code[item[1]]
            item[2] = service_code[item[2]]
            item[3] = flag_code[item[3]]
            if item[41] == "normal":
                used_data.append(item)
                item[41] = normal_code[item[41]]
        used_data = np.asarray(used_data).astype(np.float64)
        train,min_a,max_a = normalize3(used_data[:,:-2])

        print(train.shape)
        test = pd.read_csv(dataset_folder + "KDDTest+.txt", sep=',', header=None)  # 假设数据以逗号分隔
        test = test.values

        used_data = []
        for item in test:
            item[1] = protocol_type_code[item[1]]
            item[2] = service_code[item[2]]
            item[3] = flag_code[item[3]]
            if item[41] == "normal":
                item[41] = 0
            else:
                item[41] = 1
            used_data.append(item)
        used_data = np.asarray(used_data).astype(np.float64)
        train = []
        for item in used_data[:used_data.shape[0] //2, :]:
            if item[41] == 0:
                train.append(item[:-2])
        # print(train)
        train, min_a, max_a = normalize3(np.asarray(train))
        test, _, _ = normalize3(used_data[used_data.shape[0] //2:, :-2], min_a, max_a)
        labels = np.expand_dims(used_data[used_data.shape[0] //2:,41],axis=1)
        print(labels.shape,test.shape,train.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # print(train,test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
        # print(dataset, "train-len:", sum(total_train))
        # print(dataset, "test-len:", sum(total_test))
        # print(dataset, "ar:", sum(total_a) / sum(total_test))
    elif dataset == 'UCR':
        dataset_folder = 'data/UCR'
        file_list = os.listdir(dataset_folder)
        total_train = []
        total_test = []
        total_a = []
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                 dtype=np.float64,
                                 delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            print(train.shape)
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]))
            train = np.nan_to_num(train)
            test = np.nan_to_num(test)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
                np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")

        print(dataset, "train-len:", sum(total_train))
        print(dataset, "test-len:", sum(total_test))
        print(dataset, "ar:", sum(total_a) / sum(total_test))
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        total_train = []
        total_test = []
        total_a = []
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            print(train.shape)
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]))
            fn = filename.replace('.csv', '')
            train = np.nan_to_num(train)
            test = np.nan_to_num(test)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
                np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
        print(dataset, "train-len:", sum(total_train))
        print(dataset, "test-len:", sum(total_test))
        print(dataset, "ar:", sum(total_a) / sum(total_test))
    elif dataset == 'MSDS':
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        print(train)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'

        train = np.load(os.path.join(dataset_folder, 'train.npy'), allow_pickle=True)
        test = np.load(os.path.join(dataset_folder, 'test.npy'), allow_pickle=True)
        labels = np.load(os.path.join(dataset_folder, 'label.npy'), allow_pickle=True)
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        # file = os.path.join(dataset_folder, 'series.json')
        # df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        # df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.asarray(labels, dtype=np.float32)
        labels = np.nan_to_num(labels)
        # print(train)
        _, min_a, max_a = normalize3(np.concatenate((train, test), axis=0))
        train, _, _ = normalize3(train,min_a,max_a)
        test, _, _ = normalize3(test,min_a,max_a)
        print(train.shape, test.shape, labels.shape)
        print(train)
        # test, _, _ = normalize2(df_test.values)
        # labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'PSM':
        dataset_folder = 'data/PSM'
        train = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'train.csv')))[10:15000]
        test = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test.csv')))[10:15000]
        labels = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')))[10:15000]
        # train = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'train.csv')))
        # test = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test.csv')))
        # labels = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        labels = np.asarray(labels)[:, 1]
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        # file = os.path.join(dataset_folder, 'series.json')
        # df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        # df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize3(train)
        test, min_a, max_a = normalize3(test)
        print(train.shape, test.shape, labels.shape)
        print(train)
        # test, _, _ = normalize2(df_test.values)
        # labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        total_train = []
        total_test = []
        total_a = []
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')

            # train = np.pad(train, ((0, 0), (1, 1)), mode='constant', constant_values=0)
            # test  = np.pad(test, ((0, 0), (1, 1)), mode='constant', constant_values=0)
            # train, test = process_binary_columns(train, test, positive_threshold=0.15)

            train, _, _ = normalize3(train)
            test, _, _ = normalize3(test)

            # train = gaussian_filter1d(train, sigma=1.0, axis=0)
            # test = gaussian_filter1d(test, sigma=1.0, axis=0)

            # print(train)
            # if max(min_a) != max(max_a):
            #     test, _, _ = normalize3(test, min_a, max_a)
            # else:
            #     test, _, _ = normalize3(test)
            # test, _, _ = normalize3(test)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            np.savetxt(f'{folder}/{fn}_train.csv', train, delimiter=",")
            np.savetxt(f'{folder}/{fn}_test.csv', test, delimiter=",")
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]))
            train = np.nan_to_num(train)
            test = np.nan_to_num(test)
            print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]), fn)
            if fn == "C-2" or fn =="A-4":
                plotdata_with_validation(train, test, labels, dataset, fn, window_size=5, threshold=0.0001,
                                         path=folder, desc=str(fn))
                # labels = np.load(f'{dataset_folder}/test/HGCN_H-C-2_1-windows_90_epoch_1_Correctlabel.npy')
                # labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)


            # plotdata(train, test, labels,dataset, fn)
            print(fn)
            # if fn == "C-2":
            #
            #     denoising = "denoising"
            #     w = 5
            #     t = 0.0001
            #     label = plotdata_with_validation(train, test, labels, dataset, fn, window_size=5, threshold=0.0001,
            #                                      path=folder, desc=denoising)
            #     np.save(f'{folder}/{fn}_labels_-{w}-{t}_{denoising}.npy', label)
            #
            #
            #
            print(fn, np.asarray(train).shape, np.asarray(test).shape, np.asarray(labels).shape)
            np.save(f'{folder}/{fn}_labels.npy', labels)
            np.savetxt(f'{folder}/{fn}_labels.csv', labels, delimiter=",")
            print(f'{folder}/{fn}_labels.npy')
            # break

        print(dataset, "train-len:", sum(total_train))
        print(dataset, "test-len:", sum(total_test))
        print(dataset, "ar:", sum(total_a) / sum(total_test))
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1

        print(train)
        train, test, labels = convertNumpy(train), convertNumpy(test), labels[labels.columns[3:]].values[::10, :]
        # train, min_a, max_a = normalize3(train)
        # test, _, _ = normalize3(test)
        print(train)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(np.isnan(train).any(),np.isnan(test).any())
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'MBA':
        dataset_folder = 'data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
        print(train.shape, test.shape, labels.shape)
    elif dataset == 'CICIDS':
        dataset_folder = 'data/CICIDS'
        train = pd.read_csv(os.path.join(dataset_folder, '1.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        train_ori =train
        normal_data = []
        shape = train_ori.shape
        train = train_ori[:shape[0] //2, :]

        for item in train:
            if item[-1] == 'BENIGN':
                item[-1] = 0
                normal_data.append(item[:-1])


        normal_data = np.asarray(normal_data).astype(np.float64)
        # train, _, _ = normalize3(normal_data)
        train, min_a, max_a = normalize3(normal_data)
        train = np.nan_to_num(train)
        print(train.shape)

        # test = pd.read_csv(os.path.join(dataset_folder, '3.csv'))
        test = train_ori[shape[0] //2:, :]
        # test.fillna(0, inplace=True)
        # test = test.values
        normal_data = []
        for item in test:
            if item[-1] == 'BENIGN':
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        normal_data = np.asarray(normal_data).astype(np.float64)
        test, _, _ = normalize3(normal_data[:,:-1],min_a,max_a)
        test = np.nan_to_num(test)
        labels = np.expand_dims(normal_data[:,-1],axis=1)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'GAS':
        # http://www.ece.uah.edu/~thm0009/icsdatasets/gas_final.arff
        # Raw Data Gas Pipeline
        dataset_folder = 'data/GAS'
        print(dataset_folder)
        relation_name, attribute_names, data = read_arff(os.path.join(dataset_folder, 'gas_final.arff.txt'))
        # print(data)
        data = np.asarray(data).astype(np.float64)
        print(data.shape)
        normal_data = []
        for item in data:
            if item[-1] == 0:
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        data = np.asarray(normal_data)
        train = []
        train_data = data[:data.shape[0] // 2, :]
        for item in train_data:
            if item[-1] == 0:
                train.append(item[:-1])
        train, min_a, max_a = normalize3(train)
        test,_,_ =  normalize3(data[data.shape[0] //2 :, :-1])
        labels = np.expand_dims(data[data.shape[0] //2 :, -1],axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'SCADA':
        dataset_folder = 'data/SCADA'
        train = pd.read_csv(os.path.join(dataset_folder, 'ModbusRTUfeatureSetsV2/Response Injection Scrubbed V2/scrubbedWaveV2/scrubbedWaveV2.csv'))
        print(train)
        data = train.values
        print(train.columns)
        status = []
        for item in range(train.shape[1]):
            item_ele = list(set(data[:, item]))
            if len(item_ele)  < 100:
                if "Bad" in item_ele:
                    item_ele = ["Good", "Bad"]
                item_status = { item:index for index, item in enumerate(item_ele)}
                status.append(item_status)
            else:
                status.append({"code":None})
        print(status)
        for item in data:
            for i,item_ele in enumerate(status):
                if "code" not in item_ele.keys():
                    item[i] = item_ele[item[i]]

        print(data)
        train = data[:data.shape[0] //2,:]
        use_data = []
        for item in train:
            if item[-1] == 0:
                use_data.append(item[:-1])
        train,min_a,max_a = normalize3(np.asarray(use_data))
        test  =  data[data.shape[0] //2:,:-1]
        test,_,_  = normalize3(np.asarray(test),min_a,max_a)
        labels = np.expand_dims(data[data.shape[0] //2:,-1],axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'CVES':
        dataset_folder = 'data/CVES'
        # data = np.fromfile(os.path.join(dataset_folder, 's0030-04051907.dat'), dtype=np.float32)
        data = wfdb.rdrecord(os.path.join(dataset_folder, 's0030-04051907'))
        annotation = wfdb.rdann(os.path.join(dataset_folder, 's0030-04051907'), 'atr')
        data_use = data.p_signal
        print(data.p_signal.shape)
        # 打印注释信息
        print(annotation.sample)  # 标记的位置（样本点）
        # print(annotation.symbol)  # 标记的符号（如N表示正常心跳）
        # # print(annotation.time)  # 标记的详细描述
        train = data.p_signal[10000:200000,:]
        train, min_a, max_a = normalize3(train)
        test = np.concatenate([data.p_signal[8334:8534,:],data.p_signal[15872907:16072907,:],data.p_signal[16217556:16417556,:],data.p_signal[25577511:25777511,:],data.p_signal[26651098:26851098,:]],axis=0)
        labels = np.zeros(shape=data.p_signal.shape[0])
        labels[annotation.sample] = 1
        labels = np.expand_dims(labels, axis=1)
        labels = np.concatenate([labels[8334:8534,:],labels[15872907:16072907,:],labels[16217556:16417556,:],labels[25577511:25777511,:],labels[26651098:26851098,:]],axis=0)
        test,_,_ = normalize3(test,min_a,max_a)
        print(train.shape, test.shape, labels.shape)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
        # print(data_use)
    elif dataset == 'SKAB':
        dataset_folder = 'data/SKAB'
        print(dataset_folder)
        train = pd.read_csv(os.path.join(dataset_folder, 'anomaly-free/anomaly-free.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        normal_data = []
        for item in train:
            normal_data.append(item[0].split(";")[1:])
        normal_data = np.asarray(normal_data).astype(float)
        train, min_a, max_a = normalize3(normal_data)

        test = pd.read_csv(os.path.join(dataset_folder, '31.csv'))

        test.fillna(0, inplace=True)
        test = test.values[1:, :]
        normal_data = []
        for item in test:
            normal_data.append(item[0].split(";")[1:])
        normal_data = np.asarray(normal_data).astype(float)
        print(normal_data)
        test,_,_ = normalize3(normal_data[:,:-2],min_a,max_a)
        labels = np.expand_dims(normal_data[:,-2],axis=1)
        print(test)
        print(labels)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
    elif dataset == 'PowerSystem':
        dataset_folder = 'data/PowerSystem'
        print(dataset_folder)
        train = pd.read_csv(os.path.join(dataset_folder, 'data11.csv'))
        train.fillna(0, inplace=True)
        print(train)
        train = train.values
        normal_data = []
        for item in train:
           if item[-1] == 'Natural':
               normal_data.append(item[:-1])
        normal_data = np.asarray(normal_data).astype(np.float64)
        train, min_a , max_a = normalize3(normal_data)
        train = np.nan_to_num(train)

        test = pd.read_csv(os.path.join(dataset_folder, 'data12.csv'))
        test.fillna(0, inplace=True)
        print(test)
        test = test.values
        normal_data = []
        for item in test:
            if item[-1] == 'Natural':
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        normal_data = np.asarray(normal_data).astype(np.float64)
        test, _, _ = normalize3(normal_data[:,:-1],min_a,max_a)
        labels = np.expand_dims(normal_data[:,-1],axis=1)
        test = np.nan_to_num(test)
        print(train.shape, test.shape, labels.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        # train.fillna(0, inplace=True)
        # train = train.values
        # normal_data = []
        # for item in train:
        #     normal_data.append(item[0].split(";")[1:])
        # normal_data = np.asarray(normal_data).astype(float)
        # train, _, _ = normalize3(normal_data)
        #
        # test = pd.read_csv(os.path.join(dataset_folder, '31.csv'))
        #
        # test.fillna(0, inplace=True)
        # test = test.values[1:, :]
        # normal_data = []
        # for item in test:
        #     normal_data.append(item[0].split(";")[1:])
        # normal_data = np.asarray(normal_data).astype(float)
        # print(normal_data)
        # test,_,_ = normalize3(normal_data[:,:-2])
        # labels = np.expand_dims(normal_data[:,-2],axis=1)
        # print(test)
        # print(labels)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
        # print(train.shape, test.shape, labels.shape)
        # print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
    elif dataset == 'SWAN':
        dataset_folder = 'data/SWAN'
        print(dataset_folder)
        data = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_train.npy"))
        train,min_a,max_a = normalize3(data)
        test = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_test.npy"))
        test,_,_ =normalize3(test,min_a,max_a)
        labels = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_test_label.npy"))
        labels = np.repeat(np.expand_dims(labels,axis=1), test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(labels.shape, test.shape, train.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'NEGCCO':
        dataset_folder = 'data/NEGCCO'
        print(dataset_folder)

        train = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_train.npy"))
        print(train)
        # train, min_a, max_a = normalize3(train)
        test = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_test.npy"))
        _, min_a, max_a = normalize3(np.concatenate((train, test), axis=0))
        test, _, _ = normalize3(test, min_a, max_a)
        train, _, _ = normalize3(train, min_a, max_a)
        labels = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_test_label.npy"))
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        print(labels.shape, test.shape, train.shape)
        print(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]) / len(test))
        print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]))
        plotdata_with_validation(train, test, labels, dataset, dataset, window_size=5, threshold=0.0001,
                                 path=folder, desc="check_data")
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'ASD':
        dataset_folder = 'data/ASD'
        # 检测文件编码
        total_train = []
        total_test = []
        total_a = []
        for item in range(1, 13):
            train = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_train.pkl'.format(item)), "rb")))
            test = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test.pkl'.format(item)), "rb")))

            # train = np.pad(train, ((0, 0), (1, 1)), mode='constant', constant_values=0)
            # test = np.pad(test, ((0, 0), (1, 1)), mode='constant', constant_values=0)
            labels = np.asarray(
                pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test_label.pkl'.format(item)), "rb")))

            # if item == 1 :
            #     labels = np.load(os.path.join(dataset_folder, 'HGCN_H-omi-1_2-windows_5_epoch_2_Correctfinelabel.npy'.format(item)),allow_pickle=True)

            labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
            print(train.shape, test.shape, labels.shape)
            # ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
            # train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
            # test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
            train, test = train.astype(float), test.astype(float)
            # train, test = process_binary_columns(train, test, positive_threshold=0.15)
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
            np.save(os.path.join(folder, 'omi-{}_train.npy'.format(item)), train)
            np.save(os.path.join(folder, 'omi-{}_test.npy'.format(item)), test)
            np.save(os.path.join(folder, 'omi-{}_labels.npy'.format(item)), labels)
            if item == 1:
                plotdata_with_validation(train, test, labels, dataset, item, window_size=5, threshold=0.0001,
                                         path=folder, desc="check_data")
            # if item == 10:
            #     denoising = "denoising"
            #     w = 5
            #     t = 0.0001
            #     label = plotdata_with_validation(train, test, labels, dataset, item, window_size=5, threshold=0.0001,
            #                                      path=folder, desc=denoising)
            #     np.save(f'{folder}/{item}_labels_-{w}-{t}_{denoising}.npy', label)
            np.savetxt(os.path.join(folder, 'omi-{}_train.csv'.format(item)), train, delimiter=",")
            np.savetxt(os.path.join(folder, 'omi-{}_test.csv'.format(item)), test, delimiter=",")
            np.savetxt(os.path.join(folder, 'omi-{}_labels.csv'.format(item)), labels, delimiter=",")
            print("neg_num:", len(np.where(np.sum(labels, axis=1) > 0)[0]), os.path.join(dataset_folder, 'omi-{}_train.pkl'.format(item)))

            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]))
            # break

        print(dataset, "train-len:", sum(total_train))
        print(dataset, "test-len:", sum(total_test))
        print(dataset, "ar:", sum(total_a) / sum(total_test))
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    # commands = sys.argv[1:]
    load = []
    datasets = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'WADI', 'synthetic', "SCADA","PowerSystem","GAS","CICIDS","SKAB","NSL","CVES","WH","GECCO"]
    datasets =  ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'synthetic', "SCADA", "PowerSystem", "WADI", "GAS",
"CICIDS", "SKAB", "SWAN", "NEGCCO",'CVES']
    # datasets = []
    datasets = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'synthetic', "SCADA", "PowerSystem", "WADI", "GAS",
"CICIDS", "SKAB", "SWAN", "NEGCCO",'CVES']
    for item in datasets:
        print(item)
        load_data(item)
    # datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'UCR', 'MBA', 'NAB', "ASD"]
    # for d in datasets:
    #     load_data(d)
    # if len(commands) > 0:
    #     for d in commands:
    #         load_data(d)
    # else:
    #     print("Usage: python preprocess.py <datasets>")
    #     print(f"where <datasets> is space separated list of {datasets}")
