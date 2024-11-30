import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.folderconstants import *
datasets = ['SMD', 'SWaT', 'SMAP', 'MSL',"ASD","PSM"]



def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


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

def standnormalize(a,average=None,stderr=None):
    if average is None: average, stderr = np.average(a, axis=0), np.std(a, axis=0)
    return (a - average) / (stderr + 0.0001), average, stderr
def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        total_train = []
        total_test = []
        total_a = []
        for filename in file_list:
            if filename.endswith('.txt'):
                tr =load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                total_train.append(tr[0])
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                total_test.append(s[0])
                l = load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
                total_a.append(len(np.where(np.sum(np.asarray(l),axis=1) > 0)[0]))
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'

        train = np.load(os.path.join(dataset_folder, 'train.npy'), allow_pickle=True)
        test = np.load(os.path.join(dataset_folder, 'test.npy'), allow_pickle=True)
        labels = np.load(os.path.join(dataset_folder, 'label.npy'), allow_pickle=True)
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        train, min_a, max_a = normalize3(train)
        test, min_a, max_a = normalize3(test)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset == 'PSM':
        dataset_folder = 'data/PSM'
        train = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'train.csv')))[10:15000]
        test = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test.csv')))[10:15000]
        labels = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')))[10:15000]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        labels = np.asarray(labels)[:,1]
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train, min_a, max_a = normalize3(train)
        test, min_a, max_a = normalize3(test)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
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
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels),axis=1) > 0)[0]))
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'ASD':
        dataset_folder = 'data/ASD'
        # 检测文件编码
        total_train = []
        total_test = []
        total_a = []
        for item in range(1,13):
            train = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_train.pkl'.format(item)), "rb")))
            test = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test.pkl'.format(item)), "rb")))
            labels = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test_label.pkl'.format(item)), "rb")))
            labels = np.repeat(np.expand_dims(labels,axis=1),test.shape[1],axis=1)
            train, test = train.astype(float), test.astype(float)
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
            np.save(os.path.join(folder, 'omi-{}_train.npy'.format(item)),train)
            np.save(os.path.join(folder, 'omi-{}_test.npy'.format(item)),test)
            np.save(os.path.join(folder, 'omi-{}_labels.npy'.format(item)),labels)
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels),axis=1) > 0)[0]))
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
