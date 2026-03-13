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
datasets =  ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'synthetic', "SCADA", "PowerSystem", "WADI", "GAS",
"CICIDS", "SKAB", "SWAN", "NEGCCO",'CVES']
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
    # Construct the output folder path for the specific dataset
    folder = os.path.join(output_folder, dataset)
    # Create the output directory if it does not exist
    os.makedirs(folder, exist_ok=True)
    if dataset == 'synthetic':
        # Load synthetic training data and test labels
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        # Normalize training and test segments separately
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        # Initialize anomaly labels matrix
        labels = np.zeros(test.shape)
        # Mark anomaly regions in the labels matrix
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        # Inject anomalies into test data
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        # Save processed train, test, and labels to npy and csv formats
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'SMD':
        # Handle Server Machine Dataset (SMD)
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        total_train = []
        total_test = []
        total_a = []
        # Iterate through each machine file
        for filename in file_list:
            data_tr, data_te, l = [], [], []
            if filename.endswith('.txt'):
                # Load and save training data
                tr, data_tr = load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                total_train.append(tr[0])
                # Load and save test data
                s, data_te = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                total_test.append(s[0])
                # Load and save interpretation labels
                l = load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
                total_a.append(len(np.where(np.sum(np.asarray(l), axis=1) > 0)[0]))
    elif dataset == 'GECCO':
        # Handle GECCO dataset using R interop for RDS files
        dataset_folder = 'data/GECCO/'
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import rpy2py
        pandas2ri.activate()
        # Load training data from RDS
        rds_file_path = os.path.join(dataset_folder, 'waterDataTraining.RDS')
        data = robjects.r['readRDS'](rds_file_path)
        data = data.rx(True, data.colnames[1:])
        pandas_df = rpy2py(data)
        data = pandas_df.values
        # Filter out normal data for training
        use_data = []
        for item in data:
            if not item[-1]:
                use_data.append(item[:-1])
        train = np.asarray(use_data).astype(np.float64)
        train = np.nan_to_num(train)
        # Normalize training data
        train, _, _ = normalize3(train)
        # Load testing data from RDS
        rds_file_path = os.path.join(dataset_folder, 'waterDataTestingUpload.RDS')
        data = robjects.r['readRDS'](rds_file_path)
        data = data.rx(True, data.colnames[1:])
        pandas_df = rpy2py(data)
        data = pandas_df.values
        test = np.asarray(data[:, :-1]).astype(np.float64)
        test = np.nan_to_num(test)
        # Normalize test data using same parameters if needed (here independent)
        test, _, _ = normalize3(test)
        # Extract labels
        labels = np.expand_dims(data[:, -1], axis=1)
        # Clean NaN values
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Expand labels to match feature dimensions
        labels = np.repeat(labels, test.shape[1], axis=1)
        # Ensure float64 type
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        # Save processed files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'MSDS':
        # Handle MSDS dataset
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        # Downsample data by taking every 5th row
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        # Calculate normalization parameters from combined data
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        # Apply normalization
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        # Load and process labels
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        # Handle NaN values
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'SWaT':
        # Handle SWaT dataset
        dataset_folder = 'data/SWaT'
        train = np.load(os.path.join(dataset_folder, 'train.npy'), allow_pickle=True)
        test = np.load(os.path.join(dataset_folder, 'test.npy'), allow_pickle=True)
        labels = np.load(os.path.join(dataset_folder, 'label.npy'), allow_pickle=True)
        # Expand labels to match test data dimensions
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        # Handle NaN values
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.asarray(labels, dtype=np.float32)
        labels = np.nan_to_num(labels)
        # Normalize using combined min/max
        _, min_a, max_a = normalize3(np.concatenate((train, test), axis=0))
        train, _, _ = normalize3(train, min_a, max_a)
        test, _, _ = normalize3(test, min_a, max_a)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'PSM':
        # Handle PSM dataset
        dataset_folder = 'data/PSM'
        # Load and slice data
        train = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'train.csv')))[10:15000]
        test = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test.csv')))[10:15000]
        labels = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')))[10:15000]
        # Handle NaN values
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        # Process labels column
        labels = np.asarray(labels)[:, 1]
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        # Normalize data
        train, min_a, max_a = normalize3(train)
        test, min_a, max_a = normalize3(test)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset in ['SMAP', 'MSL']:
        # Handle SMAP and MSL space telemetry datasets
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        total_train = []
        total_test = []
        total_a = []
        # Process each channel file
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            # Normalize data
            train, _, _ = normalize3(train)
            test, _, _ = normalize3(test)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            np.savetxt(f'{folder}/{fn}_train.csv', train, delimiter=",")
            np.savetxt(f'{folder}/{fn}_test.csv', test, delimiter=",")
            # Generate labels from anomaly sequences
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
            np.save(f'{folder}/{fn}_labels.npy', labels)
            np.savetxt(f'{folder}/{fn}_labels.csv', labels, delimiter=",")
      
    elif dataset == 'WADI':
        # Handle WADI dataset
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        # Load train and test data with specific row skips
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        # Clean data
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        # Process time columns
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        # Map attack labels to specific columns based on time ranges
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

        # Convert to numpy and downsample
        train, test, labels = convertNumpy(train), convertNumpy(test), labels[labels.columns[3:]].values[::10, :]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'CICIDS':
        # Handle CICIDS dataset
        dataset_folder = 'data/CICIDS'
        train = pd.read_csv(os.path.join(dataset_folder, '1.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        train_ori = train
        normal_data = []
        shape = train_ori.shape
        # Split data into train and test halves
        train = train_ori[:shape[0] // 2, :]
        # Process training data (only BENIGN)
        for item in train:
            if item[-1] == 'BENIGN':
                item[-1] = 0
                normal_data.append(item[:-1])
        normal_data = np.asarray(normal_data).astype(np.float64)
        train, min_a, max_a = normalize3(normal_data)
        train = np.nan_to_num(train)

        # Process test data
        test = train_ori[shape[0] // 2:, :]
        normal_data = []
        for item in test:
            if item[-1] == 'BENIGN':
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        normal_data = np.asarray(normal_data).astype(np.float64)
        # Normalize test data using train parameters
        test, _, _ = normalize3(normal_data[:, :-1], min_a, max_a)
        test = np.nan_to_num(test)
        labels = np.expand_dims(normal_data[:, -1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Expand labels
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'GAS':
        # Handle GAS pipeline dataset
        dataset_folder = 'data/GAS'
        data = np.asarray(data).astype(np.float64)
        normal_data = []
        # Convert labels to binary
        for item in data:
            if item[-1] == 0:
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        data = np.asarray(normal_data)
        train = []
        train_data = data[:data.shape[0] // 2, :]
        # Training data contains only normal samples
        for item in train_data:
            if item[-1] == 0:
                train.append(item[:-1])
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(data[data.shape[0] // 2:, :-1])
        labels = np.expand_dims(data[data.shape[0] // 2:, -1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)

        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'SCADA':
        # Handle SCADA dataset
        dataset_folder = 'data/SCADA'
        train = pd.read_csv(os.path.join(dataset_folder, 'ModbusRTUfeatureSetsV2/Response Injection Scrubbed V2/scrubbedWaveV2/scrubbedWaveV2.csv'))
        data = train.values
        status = []
        # Encode categorical columns
        for item in range(train.shape[1]):
            item_ele = list(set(data[:, item]))
            if len(item_ele) < 100:
                if "Bad" in item_ele:
                    item_ele = ["Good", "Bad"]
                item_status = {item: index for index, item in enumerate(item_ele)}
                status.append(item_status)
            else:
                status.append({"code": None})
        for item in data:
            for i, item_ele in enumerate(status):
                if "code" not in item_ele.keys():
                    item[i] = item_ele[item[i]]
        train = data[:data.shape[0] // 2, :]
        use_data = []
        # Filter normal data for training
        for item in train:
            if item[-1] == 0:
                use_data.append(item[:-1])
        train, min_a, max_a = normalize3(np.asarray(use_data))
        test = data[data.shape[0] // 2:, :-1]
        test, _, _ = normalize3(np.asarray(test), min_a, max_a)
        labels = np.expand_dims(data[data.shape[0] // 2:, -1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'SKAB':
        # Handle SKAB dataset
        dataset_folder = 'data/SKAB'
        train = pd.read_csv(os.path.join(dataset_folder, 'anomaly-free/anomaly-free.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        normal_data = []
        # Parse semicolon-separated values
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
        test, _, _ = normalize3(normal_data[:, :-2], min_a, max_a)
        labels = np.expand_dims(normal_data[:, -2], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)

        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")

        
    elif dataset == 'PowerSystem':
        # Handle PowerSystem dataset
        dataset_folder = 'data/PowerSystem'
        train = pd.read_csv(os.path.join(dataset_folder, 'data11.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        normal_data = []
        # Filter normal data ('Natural')
        for item in train:
            if item[-1] == 'Natural':
                normal_data.append(item[:-1])
        normal_data = np.asarray(normal_data).astype(np.float64)
        train, min_a, max_a = normalize3(normal_data)
        train = np.nan_to_num(train)
        test = pd.read_csv(os.path.join(dataset_folder, 'data12.csv'))
        test.fillna(0, inplace=True)
        test = test.values
        normal_data = []
        for item in test:
            if item[-1] == 'Natural':
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        normal_data = np.asarray(normal_data).astype(np.float64)
        test, _, _ = normalize3(normal_data[:, :-1], min_a, max_a)
        labels = np.expand_dims(normal_data[:, -1], axis=1)
        test = np.nan_to_num(test)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
            
    elif dataset == 'SWAN':
        # Handle SWAN dataset
        dataset_folder = 'data/SWAN'

        data = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_train.npy"))
        train, min_a, max_a = normalize3(data)
        test = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_test.npy"))
        test, _, _ = normalize3(test, min_a, max_a)
        labels = np.load(os.path.join(dataset_folder, "NIPS_TS_Swan_test_label.npy"))
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
            
    elif dataset == 'NEGCCO':
        # Handle NEGCCO dataset
        dataset_folder = 'data/NEGCCO'
        train = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_train.npy"))
        test = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_test.npy"))
        # Calculate normalization parameters from combined data
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
        # Save files
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'ASD':
        # Handle ASD dataset (multiple files)
        dataset_folder = 'data/ASD'
        total_train = []
        total_test = []
        total_a = []
        # Iterate through 12 file parts
        for item in range(1, 13):
            train = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_train.pkl'.format(item)), "rb")))
            test = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test.pkl'.format(item)), "rb")))
            labels = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test_label.pkl'.format(item)), "rb")))
            labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
            train, test = train.astype(float), test.astype(float)
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
            # Save individual files
            np.save(os.path.join(folder, 'omi-{}_train.npy'.format(item)), train)
            np.save(os.path.join(folder, 'omi-{}_test.npy'.format(item)), test)
            np.save(os.path.join(folder, 'omi-{}_labels.npy'.format(item)), labels)
            np.savetxt(os.path.join(folder, 'omi-{}_train.csv'.format(item)), train, delimiter=",")
            np.savetxt(os.path.join(folder, 'omi-{}_test.csv'.format(item)), test, delimiter=",")
            np.savetxt(os.path.join(folder, 'omi-{}_labels.csv'.format(item)), labels, delimiter=",")
            total_train.append(len(train))
            total_test.append(len(test))
            total_a.append(len(np.where(np.sum(np.asarray(labels), axis=1) > 0)[0]))

    else:
        # Raise exception if dataset is not supported
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    commands = sys.argv[1:]
    if len(commands) > 0:
        for d in commands:
            load_data(d)
            print(f'Processed {d} is saved in {os.path.join(output_folder, d)}')
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
