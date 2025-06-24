import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.folderconstants import *

datasets = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'Synthetic', 'GPT', 'PowerSystem', 'WADI',
            'GasPipeline', 'CICIDS', 'SKAB', 'SWAN', 'GECCO']


def load_and_save(category, filename, dataset, dataset_folder):
    """
    Loads a CSV file containing dataset samples, saves it as a `.npy` file for faster loading later,
    and returns the shape of the loaded data.

    This function is primarily used for preprocessing time-series datasets like SMD (Server Machine Dataset),
    where raw data is stored in CSV format and needs to be converted into NumPy arrays for training or testing.

    Parameters:
    - category (str): The type of data being loaded; typically 'train', 'test', or similar.
    - filename (str): Name of the input CSV file containing the data.
    - dataset (str): Identifier for the specific dataset being processed.
    - dataset_folder (str): Path to the folder containing the dataset files organized by category.

    Returns:
    - tuple: Shape of the loaded data array (`temp.shape`), indicating number of samples and features.
    """

    # Construct full path to the CSV file and load its contents into a NumPy array
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')

    # Print progress information about the dataset being processed
    print(dataset, category, filename, temp.shape)

    # Save the loaded data as a `.npy` file for future fast access
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

    # Return the dimensions of the loaded data (number of rows and columns)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    """
    Parses anomaly label files and generates a binary label matrix indicating anomalous regions.

    This function reads label files that define time ranges and specific features affected by anomalies.
    It constructs a binary matrix (`temp`) where entries are set to 1 if an anomaly is present at that
    time step and feature index. The resulting matrix is saved as a `.npy` file for later use in training/testing.

    Parameters:
    - category (str): Type of data being processed (e.g., 'labels').
    - filename (str): Name of the label file containing anomaly descriptions.
    - dataset (str): Identifier for the specific dataset being processed.
    - dataset_folder (str): Path to the folder containing dataset files.
    - shape (tuple): Shape of the output label matrix (typically same as corresponding test data).

    Returns:
    - temp (np.ndarray): A NumPy array representing the binary labels with shape `shape`.
    """

    # Initialize an empty matrix of given shape to store binary labels
    temp = np.zeros(shape)

    # Open and read the label file line by line
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()

    # Process each line to extract anomaly intervals and affected feature indices
    for line in ls:
        # Split line into position range and feature indices
        pos, values = line.split(':')[0], line.split(':')[1].split(',')

        # Parse start and end positions of the anomaly interval
        start, end = int(pos.split('-')[0]), int(pos.split('-')[1])

        # Convert feature indices to zero-based indexing
        indx = [int(i) - 1 for i in values]

        # Mark all time steps within the interval and affected features as anomalous (1)
        temp[start - 1:end - 1, indx] = 1

    # Print progress information
    print(dataset, category, filename, temp.shape)

    # Save the generated label matrix as a `.npy` file
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

    # Return the generated label matrix
    return temp


def normalize(a):
    """
    Normalizes the input array `a` to the range [-1, 1] based on the max absolute value,
    and then scales it to the range [0, 1].

    This normalization is feature-wise (column-wise), meaning each feature (sensor/channel)
    is scaled independently.

    Parameters:
    - a (np.ndarray): Input array of shape (num_samples, num_features).

    Returns:
    - np.ndarray: Normalized array with values in the range [0, 1].
    """

    # Normalize each feature by its maximum absolute value to get into range [-1, 1]
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))

    # Scale from [-1, 1] to [0, 1]
    return (a / 2 + 0.5)


def normalize2(a, min_a=None, max_a=None):
    """
    Performs Min-Max normalization on the input array `a`, scaling values to the range [0, 1].

    If min and max values are not provided, they are computed from the data. The normalization
    is done per-feature (column-wise) based on the minimum and maximum values.

    Parameters:
    - a (array-like): Input data array (list or numpy array).
    - min_a (float or None): Optional minimum value for normalization. If None, computed from data.
    - max_a (float or None): Optional maximum value for normalization. If None, computed from data.

    Returns:
    - tuple:
        - normalized_data (np.ndarray): Array with values scaled to [0, 1].
        - min_a (float): Minimum value used in normalization.
        - max_a (float): Maximum value used in normalization.
    """

    # If min and max are not provided, compute them from the data
    if min_a is None:
        min_a, max_a = min(a), max(a)

    # Apply Min-Max scaling: (x - min) / (max - min)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    """
    Performs feature-wise Min-Max normalization on the input array `a`, scaling each feature to the range [0, 1].

    If min and max values are not provided, they are computed from the data along feature axis (column-wise).
    A small epsilon (0.0001) is added to the denominator to prevent division by zero.

    Parameters:
    - a (np.ndarray): Input data array of shape (num_samples, num_features).
    - min_a (np.ndarray or None): Optional minimum values for each feature. If None, computed from data.
    - max_a (np.ndarray or None): Optional maximum values for each feature. If None, computed from data.

    Returns:
    - tuple:
        - normalized_data (np.ndarray): Array with each feature scaled to [0, 1].
        - min_a (np.ndarray): Minimum values used for normalization (per feature).
        - max_a (np.ndarray): Maximum values used for normalization (per feature).
    """

    # If min and max are not provided, compute them per feature (column-wise)
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)

    # Apply Min-Max scaling with small epsilon to avoid division by zero
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def standnormalize(a, average=None, stderr=None):
    """
    Performs feature-wise standardization (Z-score normalization) on the input array `a`.

    Each feature (column) is normalized to have zero mean and unit variance.
    If mean (`average`) and standard deviation (`stderr`) are not provided,
    they are computed from the data.

    Parameters:
    - a (np.ndarray): Input data array of shape (num_samples, num_features).
    - average (np.ndarray or None): Optional mean values for each feature. If None, computed from data.
    - stderr (np.ndarray or None): Optional standard deviation values for each feature. If None, computed from data.

    Returns:
    - tuple:
        - normalized_data (np.ndarray): Array with each feature standardized.
        - average (np.ndarray): Mean values used for normalization (per feature).
        - stderr (np.ndarray): Standard deviation values used for normalization (per feature).
    """

    # If mean and std are not provided, compute them per feature (column-wise)
    if average is None:
        average, stderr = np.average(a, axis=0), np.std(a, axis=0)

    # Apply Z-score normalization with small epsilon to avoid division by zero
    return (a - average) / (stderr + 0.0001), average, stderr


def convertNumpy(df):
    """
    Converts a pandas DataFrame to a NumPy array and applies downsampling and Min-Max normalization.

    This function selects all columns starting from the 4th column (index 3),
    then takes every 10th row to reduce data size (downsampling). The selected data is
    then normalized to the range [0, 1] using Min-Max scaling.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series or sensor data.

    Returns:
    - np.ndarray: Downsampled and normalized array of shape (num_samples // 10, num_features).
    """

    # Select columns starting from the 4th column and downsample by taking every 10th row
    x = df[df.columns[3:]].values[::10, :]

    # Apply Min-Max normalization: (x - min) / (max - min + epsilon)
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    """
    Preprocesses and saves time series data for various anomaly detection datasets.

    This function handles the loading, normalization, labeling, and saving of multiple datasets.
    Each dataset has specific preprocessing logic including normalization, downsampling,
    label generation, and file I/O operations.

    Supported Datasets:
        - synthetic: Synthetic time series with injected anomalies
        - SMD (Server Machine Dataset): Real-world server monitoring logs
        - MSDS: Multi-sensor dataset
        - SWAT: Cyber-physical water treatment testbed dataset
        - PSM: Plant Simulation Model dataset
        - SMAP & MSL: NASA telemetry sensor data
        - WADI: Water Distribution Industrial Control System dataset
        - CICIDS: Network intrusion detection dataset
        - GasPipeline: Simulated gas pipeline dataset
        - GPT: Modbus protocol traffic dataset
        - SKAB: Sensor-KPI Anomaly Benchmark
        - PowerSystem: Electrical power system measurements
        - SWAN: Spatiotemporal Water Network dataset
        - GECCO: Environmental sensor data from GECCO competition
        - ASD: Autism Spectrum Disorder behavioral data

    Parameters:
        dataset (str): Name of the dataset to process. Must be one of the predefined values.

    Output:
        Creates a folder under `output_folder` named after the dataset.
        Saves preprocessed data in both `.npy` and `.csv` formats:
            - train.npy/csv: Normalized training data
            - test.npy/csv: Normalized test data
            - labels.npy/csv: Binary anomaly labels aligned with test data

    Raises:
        Exception: If an unsupported dataset name is provided.
    """
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
            if filename.endswith('.txt'):
                tr = load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                total_train.append(tr[0])
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                total_test.append(s[0])
                l = load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
                total_a.append(len(np.where(np.sum(np.asarray(l), axis=1) > 0)[0]))
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

        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'
        train = np.load(os.path.join(dataset_folder, 'train.npy'), allow_pickle=True)
        test = np.load(os.path.join(dataset_folder, 'test.npy'), allow_pickle=True)
        labels = np.load(os.path.join(dataset_folder, 'label.npy'), allow_pickle=True)
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        _, min_a, max_a = normalize3(np.concatenate((train, test), axis=0))
        train, _, _ = normalize3(train, min_a, max_a)
        test, _, _ = normalize3(test, min_a, max_a)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'PSM':
        dataset_folder = 'data/PSM'
        train = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'train.csv')))[10:15000]
        test = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test.csv')))[10:15000]
        labels = np.asarray(pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')))[10:15000]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.nan_to_num(labels)
        labels = np.asarray(labels)[:, 1]
        labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
        train, min_a, max_a = normalize3(train)
        test, min_a, max_a = normalize3(test)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
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
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
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
            np.save(f'{folder}/{fn}_labels.npy', labels)
            np.savetxt(f'{folder}/{fn}_labels.csv', labels, delimiter=",")
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
        train, test, labels = convertNumpy(train), convertNumpy(test), labels[labels.columns[3:]].values[::10, :]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'CICIDS':
        dataset_folder = 'data/CICIDS'
        train = pd.read_csv(os.path.join(dataset_folder, '1.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        train_ori = train
        normal_data = []
        shape = train_ori.shape
        train = train_ori[:shape[0] // 2, :]
        for item in train:
            if item[-1] == 'BENIGN':
                item[-1] = 0
                normal_data.append(item[:-1])
        normal_data = np.asarray(normal_data).astype(np.float64)
        train, min_a, max_a = normalize3(normal_data)
        train = np.nan_to_num(train)
        test = train_ori[shape[0] // 2:, :]
        normal_data = []
        for item in test:
            if item[-1] == 'BENIGN':
                item[-1] = 0
            else:
                item[-1] = 1
            normal_data.append(item)
        normal_data = np.asarray(normal_data).astype(np.float64)
        test, _, _ = normalize3(normal_data[:, :-1], min_a, max_a)
        test = np.nan_to_num(test)
        labels = np.expand_dims(normal_data[:, -1], axis=1)

        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'GasPipeline':
        dataset_folder = 'data/GasPipeline'
        relation_name, attribute_names, data = read_arff(os.path.join(dataset_folder, 'gas_final.arff.txt'))
        data = np.asarray(data).astype(np.float64)
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
        test, _, _ = normalize3(data[data.shape[0] // 2:, :-1])
        labels = np.expand_dims(data[data.shape[0] // 2:, -1], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'GPT':
        dataset_folder = 'data/GPT'
        train = pd.read_csv(os.path.join(dataset_folder,
                                         'ModbusRTUfeatureSetsV2/Response Injection Scrubbed V2/scrubbedWaveV2/scrubbedWaveV2.csv'))
        data = train.values
        status = []
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
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'SKAB':
        dataset_folder = 'data/SKAB'
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
        test, _, _ = normalize3(normal_data[:, :-2], min_a, max_a)
        labels = np.expand_dims(normal_data[:, -2], axis=1)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        labels = np.repeat(labels, test.shape[1], axis=1)
        train = train.astype(np.float64)
        test = test.astype(np.float64)
        labels = labels.astype(np.float64)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'PowerSystem':
        dataset_folder = 'data/PowerSystem'
        train = pd.read_csv(os.path.join(dataset_folder, 'data11.csv'))
        train.fillna(0, inplace=True)
        train = train.values
        normal_data = []
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
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file).astype('float64'), delimiter=",")
    elif dataset == 'SWAN':
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
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'GECCO':
        dataset_folder = 'data/GECCO'
        train = np.load(os.path.join(dataset_folder, "NIPS_TS_Water_train.npy"))
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
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
            np.savetxt(os.path.join(folder, f'{file}.csv'), eval(file), delimiter=",")
    elif dataset == 'ASD':
        dataset_folder = 'data/ASD'
        total_train = []
        total_test = []
        total_a = []
        for item in range(1, 13):
            train = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_train.pkl'.format(item)), "rb")))
            test = np.asarray(pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test.pkl'.format(item)), "rb")))
            labels = np.asarray(
                pickle.load(open(os.path.join(dataset_folder, 'omi-{}_test_label.pkl'.format(item)), "rb")))
            labels = np.repeat(np.expand_dims(labels, axis=1), test.shape[1], axis=1)
            train, test = train.astype(float), test.astype(float)
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test)
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
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    """
    Entry point of the script when executed from the command line.

    This section handles command-line arguments and triggers the `load_data` function 
    for each specified dataset. If no datasets are provided, it prints a usage message.

    Command Line Usage:
        python preprocess.py <dataset1> <dataset2> ...

    Example:
        python preprocess.py SMD PSM

    Output:
        Processes and saves normalized training and test data along with labels
        for each specified dataset into the output directory defined in `output_folder`.
    """
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
