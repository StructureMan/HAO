
import gc
import glob
import random
import traceback
import warnings
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from src import models  
import src.models
from src.evaluate import *
from src.merlin import *
from visualData.visualData import plotPried
from utils.data_utils import *
import time
device = "cuda:0"  
GLOBAL_SEED = 666
warnings.filterwarnings("ignore")


def convert_to_windows(data, model):
    """
    Converts input data into sliding windows for model training or testing.

    Args:
        data (torch.Tensor): Input time series data with shape (num_samples, num_features).
        model (object): Model object that must have an attribute [n_window] specifying the window size.

    Returns:
        torch.Tensor: Sliding window data with shape (num_windows, window_size, num_features) 
                      or (num_windows, window_size * num_features), depending on the model type.

    Functionality:
        1. Slices the input data into sliding windows based on the model's window size (`model.n_window`).
        2. For each time step, if the current index is less than the window size, 
           the initial part is padded using the first sample.
        3. Depending on the model name, the window data is either kept in its original structure 
           or flattened:
           - Specific models (e.g., "ST_GSLN", "GSL_AD", etc.) retain the original window structure.
           - Other models (e.g., "TranAD", "Attention") flatten the window data conditionally.
    """
    windows = []  # Stores generated window data
    w_size = model.n_window  # Get the window size defined by the model

    for i, g in enumerate(data):  # Iterate over each row of the input data
        if i >= w_size:  # If current index is greater than or equal to window size
            w = data[i - w_size:i]  # Directly slice the window-sized data
        else:  # Otherwise, pad the initial part
            # Repeat the first sample and concatenate with current data
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])

        # Decide how to process window data based on model type
        if args.model in ["DGINet"]:
            windows.append(w)  # Specific models retain window structure
        else:
            # Other models conditionally flatten window data
            windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))

    return torch.stack(windows)  # Stack all windows into a tensor and return


class dataLoader(Dataset):

    def __init__(self, DataName="", modelName="", convertWindow="", stage="Train", item_dataSet="", path=None):
        """
        Initializes the dataLoader class for loading and preprocessing datasets.

        Args:
            DataName (str): Name of the dataset (e.g., 'ASD', 'SMD').
            modelName (str): Name of the model being used.
            convertWindow (int): Window size for sliding window conversion.
            stage (str): Stage of the process ('Train' or 'Test').
            item_dataSet (str): Specific subset of the dataset.
            path (str, optional): Path to custom label data if provided.
        """
        self.DataName = DataName
        self.modelName = modelName
        self.convertWindow = convertWindow
        self.stage = stage

        # Models that require flattening of sliding windows
        self.limitModel = [
                           'MSCRED', 'CAE_M', 'MTAD_GAT', "GRN", "STADN", "TS_GAT", "DGINet", "GDN", 'MAD_GAN', "TranAD",
                           "HAO_E", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_P_MSCD", 
                           "HAO_P", "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_E_MSCD",
                           "HAO_H", "HAO_H_HDNN",  "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD", "HAO_H_MSCD",                       
                           ]

        # Models that do not require flattening of sliding windows
        self.limitStackModel = ["DGINet"]
        
        # Models that use default window settings
        exception_models = ["GDN", "GRN", "STADN", "TS_GAT","MAD_GAN","TranAD", "DGINet", 
                            "HAO_E", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_P_MSCD", 
                            "HAO_P", "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_E_MSCD",
                            "HAO_H", "HAO_H_HDNN",  "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD", "HAO_H_MSCD",
                            ]
        
        folder = os.path.join(output_folder, self.DataName)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        
        self.loader = []
        for file in ['train', 'test', 'labels']:
            if path is not None and file == "labels":
                file = path
                self.loader.append(np.load(file, allow_pickle=True))
            else:
                if DataName == 'ASD': file = item_dataSet + file
                if DataName == 'SMD': file = item_dataSet + file
                if DataName == 'SMAP': file = item_dataSet + file
                if DataName == 'MSL': file = item_dataSet + file
                if DataName == 'UCR': file = item_dataSet + file
                if DataName == 'NAB': file = item_dataSet + file
                self.loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True))

        # Check if the data is extremely sparse
        self.is_change_branch = 3 in np.sort(list(set(np.sum((self.loader[0] > 0.999) + 0, axis=0))))
        self.labes_ture = torch.tensor((np.sum(self.loader[2], axis=1) >= 1) + 0)
        self.labels_counts = torch.bincount(self.labes_ture)
        self.class_weights = 1.0 / self.labels_counts.float()
        self.sample_weights = self.class_weights[torch.tensor(self.labes_ture)]

        if self.modelName in exception_models:
            self.convertWindow = convertWindow
        else:
            self.convertWindow = self.getDim()
        
        if self.modelName in self.limitModel:
            self.trainData = self.convertWindowPro(0)
            self.testData = self.convertWindowPro(1)
            self.labelData = self.convertWindowPro(2)

    def getDim(self):
        """Returns the number of features in the dataset."""
        return self.loader[0].shape[1]

    def convertWindowPro(self, index):
        """
        Converts data into sliding windows for specific models.

        Args:
            index (int): Index of the data split (0: train, 1: test, 2: labels).

        Returns:
            torch.Tensor: Sliding window data.
        """
        if self.modelName in self.limitModel:
            windows = []
            data = torch.Tensor(self.loader[index])
            for i, g in enumerate(self.loader[index]):
                if i >= self.convertWindow:
                    w = data[i - self.convertWindow:i]
                else:
                    w = torch.cat([data[0].repeat(self.convertWindow - i, 1), data[0:i]])
                
                if self.modelName in self.limitStackModel:
                    windows.append(w)
                else:
                    windows.append(
                        w if 'TranAD' in self.modelName or 'Attention' in self.modelName else w.contiguous().view(-1))
            return torch.stack(windows)

    def __len__(self):
        """
        Returns the length of the dataset based on the stage and model requirements.

        Returns:
            int: Length of the dataset.
        """
        if self.stage == "Train":
            if self.modelName in self.limitModel:
                return len(self.trainData) - 1
            else:
                return len(self.loader[0]) - 1
        else:
            if self.modelName in self.limitModel:
                return len(self.testData) - 1
            else:
                return len(self.loader[1]) - 1


    def __getitem__(self, item):
        """
        Retrieves a specific item from the dataset based on the stage and model requirements.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            tuple or torch.Tensor: Data item(s) depending on the stage and model.
        """
        if self.stage == "Train":
            if self.modelName in self.limitModel:
                return torch.FloatTensor(self.trainData[item]).to(device).to(torch.float64)
            else:
                return torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)
        else:
            if self.modelName in self.limitModel:
                return torch.FloatTensor(self.testData[item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.labelData[item]).to(
                    device).to(torch.float64), torch.FloatTensor(self.loader[2][item]).to(device).to(torch.float64)
            else:
                return torch.FloatTensor(self.loader[1][item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.loader[2][item]).to(
                    device).to(torch.float64), torch.FloatTensor(self.loader[2][item]).to(device).to(torch.float64)


def load_dataset(dataset, is_corr):
    """
    Loads and preprocesses the dataset for training and testing.

    Args:
        dataset (str): Name of the dataset to load (e.g., 'ASD', 'SMD', 'SMAP', etc.).
        is_corr (bool): Flag indicating whether to apply correlation-based preprocessing.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training data.
            - test_loader (DataLoader): DataLoader for the testing data.
            - labels (np.ndarray): Labels for the test data (excluding the first element).

    Functionality:
        1. Constructs the folder path for the specified dataset.
        2. Checks if the processed data exists; raises an exception if not found.
        3. Loads the 'train', 'test', and 'labels' files for the dataset.
        4. Applies optional data reduction if `args.less` is enabled.
        5. Creates DataLoader objects for training and testing data.
        6. Returns the DataLoaders and test labels for further processing.
    """
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'ASD': file = 'omi-1_' + file
        if dataset == 'SMD': file = 'machine-3-7_' + file
        if dataset == 'SMAP': file = 'A-4_' + file
        if dataset == 'MSL': file = 'C-2_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))

    if args.less: 
        loader[0] = cut_array(0.2, loader[0])  # Reduce training data if specified
    
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0]-1)
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0]-1)
    labels = loader[2][1:]  # Exclude the first label entry

    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, n_windows, item_name, batch=None, desc="", epoch=0):
    """
    Saves the trained model's state dictionary to a specified file path.

    Args:
        model (torch.nn.Module): The trained model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): Optimizer used during training (optional).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler used during training (optional).
        n_windows (int): Number of windows used in the model.
        item_name (str): Name of the dataset item or subset.
        batch (int, optional): Batch size used during training.
        desc (str, optional): Description or identifier for the experiment.
        epoch (int, optional): Current epoch number.

    Functionality:
        1. Constructs a folder path based on the experiment description, number of windows, 
           model name, dataset, item name, and batch size.
        2. Creates the directory if it does not already exist.
        3. Saves the model's state dictionary to a `.ckpt` file within the created folder.
        4. Optionally includes optimizer and scheduler states in the saved file (commented out by default).
    """
    folder = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
    }, file_path)


def load_model(modelname, dims, n_windows, batch=None, item_name="", desc="", epoch=0, is_change_branch=True):
    """
    Loads a pre-trained model or creates a new one based on the provided parameters.

    Args:
        modelname (str): Name of the model to load or create.
        dims (int): Dimensionality of the input data.
        n_windows (int): Number of windows used in the model.
        batch (int, optional): Batch size used during training.
        item_name (str, optional): Name of the dataset item or subset.
        desc (str, optional): Description or identifier for the experiment.
        epoch (int, optional): Current epoch number.
        is_change_branch (bool, optional): Flag indicating whether to change the branch structure.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): Loaded or newly created model.
            - optimizer (torch.optim.Optimizer): Optimizer for the model.
            - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            - is_history (bool): Indicates whether a pre-trained model was loaded.

    Functionality:
        1. Dynamically retrieves the model class based on `modelname`.
        2. Instantiates the model with appropriate parameters:
           - For specific models (e.g., "HAO"), additional parameters like `n_windows` and [is_change_branch] are used.
           - For other models, only `dims` is passed.
        3. Initializes the optimizer (AdamW) and learning rate scheduler (StepLR).
        4. Attempts to load a pre-trained model from a checkpoint file if it exists and retraining/testing is not forced.
        5. If no pre-trained model is found or retraining is requested, a new model is created.
        6. Handles exceptions gracefully and prints error messages if any occur.
    """
    model = None
    optimizer = None
    scheduler = None
    try:
        # Retrieve the model class dynamically
        model_class = getattr(src.models, modelname)
        
        # Instantiate the model based on its type
        if modelname in ["HAO_E", "HAO_P", "HAO_H", "DGINet", "HAO_E_HDNN",
                         "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD",
                         "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_E_MSCD", "HAO_P_MSCD",
                         "HAO_H_MSCD", "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
            model = model_class(dims, n_windows, is_change_branch).double()
        else:
            model = model_class(dims).double()

        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        
        # Determine if a pre-trained model should be loaded
        is_history = None
        fname = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/model.ckpt'
        if os.path.exists(fname) and (not args.retrain or args.test):
            print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            is_history = True
        else:
            print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
            is_history = False
    except Exception as e:
        print(e)
    return model, optimizer, scheduler, is_history


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True, adj_save=None, up_threshould=0.0,
             data_std=0.0, is_search=False, check_list=None):
    """
    Performs forward and backward propagation for training or testing a model.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to be trained or tested.
        data (torch.Tensor): Input data for the model.
        dataO (int): Number of features in the input data.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        training (bool, optional): Whether the model is in training mode. Defaults to True.
        adj_save (list, optional): Paths for saving adjacency matrices and training logs.
        up_threshould (float, optional): Threshold for adjacency matrix updates.
        data_std (float, optional): Standard deviation multiplier for anomaly detection thresholds.
        is_search (bool, optional): Whether to perform hyperparameter search.
        check_list (list, optional): Precomputed adjacency matrices and predictions for reuse.

    Returns:
        tuple: Depending on the mode (training/test) and model type, returns:
               - Training: (loss, learning_rate)
               - Testing: (loss, predictions, additional_outputs, original_data)
    """
    global loss
    l = nn.MSELoss(reduction='mean' if training else 'none')  # Loss function
    feats = dataO  # Number of features

    # Handle OmniAnomaly-specific logic
    if 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            total_data = len(data)
            for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                y_pred, mu, logvar, hidden = model(d.to(torch.float64).to(device), hidden.to(device) if i else None)
                MSE = l(y_pred.to(torch.float64).to(device), d.to(torch.float64).to(device))
                KLD = -0.5 * torch.sum(1 + logvar.to(torch.float64) - mu.pow(2) - logvar.to(torch.float64).exp(), dim=0)
                loss = MSE.to(torch.float64) + model.beta * KLD.to(torch.float64)
                mses.append(torch.mean(MSE.to(torch.float64)).item())
                klds.append(model.beta * torch.mean(KLD.to(torch.float64)).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            total_data = len(data)
            for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                y_pred, _, _, hidden = model(d.to(device), None)
                y_preds.append(y_pred.to(torch.float64))
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred.to(torch.float64).to(device), data.to(torch.float64).to(device))
            return MSE.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), None, data.to(device)

    # Handle GDN, MTAD_GAT, and related models
    elif model.name in ["GDN", "MTAD_GAT", "MSCRED", "CAE_M", "GRN", "TS_GAT", "STADN",
                        "HAO_E", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_E_MSCD",
                        "HAO_P", "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_P_MSCD",
                        "HAO_H", "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD", "HAO_H_MSCD"]:
        l = nn.MSELoss(reduction='none')
        l_s = nn.MSELoss(reduction='none')
        n = epoch + 1
        l1s = []

        if training:
            # Training logic for HAO-related models
            if model.name in ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN",
                              "HAO_E_AHGSD", "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                              "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD",
                              "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
                # Initialize training statistics dictionary
                save_curv_dict = {
                    "epoch": [],
                    "s_liner_curv": [],
                    "t_liner_curv": [],
                    "t_HGCN_in_curv": [],
                    "s_HGCN_in_curv": [],
                    "t_HGCN_out_curv": [],
                    "s_HGCN_out_curv": [],
                    "rec_loss": [],
                    "T_loss": [],
                    "S_loss": [],
                    "loss": [],
                    "time": [],
                }
                try:
                    with open(adj_save[2], 'r', encoding='utf-8') as f:
                        save_curv_dict = json.load(f)
                except Exception as e:
                    pass
                epoch = []
                losss = torch.tensor([0]).to(device).to(torch.float64)
                rec_loss = torch.tensor([0]).to(device).to(torch.float64)
                T_loss = torch.tensor([0]).to(device).to(torch.float64)
                S_loss = torch.tensor([0]).to(device).to(torch.float64)
                t_adj_q = None
                s_adj_q = None
                t_adj = None
                s_adj = None
                t_adj_list = []
                s_adj_list = []
                history_t = None
                history_s = None
                try:
                    t_adj_q = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
                    s_adj_q = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
                except Exception as e:
                    pass
                curv = None
                pbar = tqdm(enumerate(data), total=len(data), desc=f'Epoch {epoch}, Loss{losss.item()}')
                for i, d in pbar:
                    start_time = time.time()
                    if model.name in ["MTAD_GAT"]:
                        x, h = model(d, h if i else None)
                    elif model.name in ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN",
                                        "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                                        "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD",
                                        "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
                        x, s_adj_q, t_adj_q, curv, t_adj, s_adj = model(d, t_adj_q, s_adj_q)
                        if len(t_adj_q.shape) > 2:
                            t_adj_q = t_adj_q.squeeze(dim=2)
                            s_adj_q = s_adj_q.squeeze(dim=2)
                        t_adj_list.append(t_adj_q)
                        s_adj_list.append(s_adj_q)
                    else:
                        x = model(d)

                    # Update curvature and adjacency matrices
                    if 'curv' in locals() or 'curv' in globals():
                        curv_list = curv.cpu().detach().numpy()
                        save_curv_dict["s_liner_curv"].append(float(curv_list[0][0]))
                        save_curv_dict["epoch"].append(float(len(save_curv_dict["epoch"])))
                        save_curv_dict["t_liner_curv"].append(float(curv_list[1][0]))
                        save_curv_dict["t_HGCN_in_curv"].append(float(curv_list[2][0]))
                        save_curv_dict["s_HGCN_in_curv"].append(float(curv_list[3][0]))
                        save_curv_dict["t_HGCN_out_curv"].append(float(curv_list[4][0]))
                        save_curv_dict["s_HGCN_out_curv"].append(float(curv_list[5][0]))

                    if len(t_adj_list) < 2:
                        history_t = t_adj_list[0]
                        history_s = s_adj_list[0]
                    else:
                        history_t = t_adj_list[0] - (t_adj_list[0] - t_adj_list[1]) / i
                        history_s = s_adj_list[0] - (s_adj_list[0] - s_adj_list[1]) / i

                    t_adj_list = [nn.Parameter(history_t, requires_grad=False).to(device).to(torch.float64)]
                    s_adj_list = [nn.Parameter(history_s, requires_grad=False).to(device).to(torch.float64)]
                    T_loss = torch.mean(l_s(history_t, t_adj_q))
                    rec_loss = torch.mean(l(x, d))
                    if T_loss.cpu().detach().numpy() == 0:
                        T_loss = nn.Parameter(torch.Tensor([1.0])).to(device).to(torch.float64)
                        S_loss = nn.Parameter(torch.Tensor([1.0])).to(device).to(torch.float64)
                        losss = rec_loss
                    else:
                        T_loss = torch.mean(l_s(history_t, t_adj_q))
                        S_loss = torch.mean(l_s(history_s, s_adj_q))
                        losss = 0.8 * rec_loss + 0.1 * T_loss + 0.1 * S_loss
                        grad_T = torch.autograd.grad(T_loss, t_adj_q, retain_graph=True)[0]
                        grad_S = torch.autograd.grad(S_loss, s_adj_q, retain_graph=True)[0]
                        t_adj_q = history_t - optimizer.param_groups[0]['lr'] * grad_T
                        s_adj_q = history_s - optimizer.param_groups[0]['lr'] * grad_S

                    # Save training statistics
                    if 'save_curv_dict' in locals() or 'save_curv_dict' in globals():
                        save_curv_dict["rec_loss"].append(float(rec_loss.cpu().detach().numpy()))
                        save_curv_dict["T_loss"].append(float(T_loss.cpu().detach().numpy()))
                        save_curv_dict["S_loss"].append(float(S_loss.cpu().detach().numpy()))
                        save_curv_dict["loss"].append(float(losss.cpu().detach().numpy()))
                    save_curv_dict["time"].append(time.time() - start_time)

                    l1s.append(losss.item())
                    optimizer.zero_grad()
                    losss.backward()
                    optimizer.step()
                    max_norm = float(1)
                    all_params = list(model.parameters())
                    for param in all_params:
                        torch.nn.utils.clip_grad_norm_(param, max_norm)

                    pbar.set_description(
                        desc=f'Epoch {i}, Loss{losss.item():.8f}')
                np.save(adj_save[0], history_t.cpu().detach().numpy())
                np.save(adj_save[1], history_s.cpu().detach().numpy())
                with open(adj_save[2], 'w', encoding='utf-8') as f:
                    json.dump(save_curv_dict, f, ensure_ascii=False, indent=4)

                try:
                    del save_curv_dict
                    gc.collect()
                except Exception as e:
                    print(e)
                    print(traceback.print_exc())
                return np.mean(l1s), optimizer.param_groups[0]['lr']
            else:
                # Training logic for other models
                for i, d in enumerate(data):
                    times = time.time()
                    if model.name in ["MTAD_GAT"]:
                        x, h = model(d, h if i else None)
                    else:
                        x = model(d.to(device))
                    loss = torch.mean(l(x, d))
                    l1s.append(torch.mean(loss).item())
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # Testing logic
            if model.name in ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN",
                              "HAO_E_AHGSD", "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                              "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD",
                              "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
                xs = []
                adj_list_t = []
                adj_list_s = []
                jacced = []
                # Load pre-trained adjacency matrices
                t_adj = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
                s_adj = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
                t_len, t_tri_len = t_adj.shape[0], (t_adj.shape[0] * (t_adj.shape[0] - 1)) // 2
                s_len, s_tri_len = s_adj.shape[0], (s_adj.shape[0] * (s_adj.shape[0] - 1)) // 2
                save_1 = t_adj.cpu().detach().numpy()
                save_1 *= np.triu(np.ones((t_len, t_len))) - np.eye(t_len)
                threshold_value = np.partition(save_1, -t_tri_len, axis=None)[-t_tri_len]
                save_1[save_1 < threshold_value] = 0
                index_s = [f"timestamp:{item}" for item in range(t_len)]
                df = pd.DataFrame(save_1, index=index_s, columns=index_s)
                df.to_csv(adj_save[3])
                df = pd.DataFrame(save_1 + save_1.T, index=index_s, columns=index_s)
                df.to_csv(adj_save[5])
                save_1 = s_adj.cpu().detach().numpy()
                save_1 *= np.triu(np.ones((s_len, s_len))) - np.eye(s_len)
                threshold_value = np.partition(save_1, -s_tri_len, axis=None)[-s_tri_len]
                save_1[save_1 < threshold_value] = 0
                index_s = [f"sensor:{item}" for item in range(s_len)]
                df = pd.DataFrame(save_1, index=index_s, columns=index_s)
                df.to_csv(adj_save[4])
                df = pd.DataFrame(save_1 + save_1.T, index=index_s, columns=index_s)
                df.to_csv(adj_save[6])
                data_shape = torch.zeros(size=data[0].shape).view(-1, feats).shape
                total_data = len(data)
                fine_lable = torch.zeros(size=(total_data, data_shape[0], data_shape[1]))

                if check_list is None:
                    s_list = []
                    t_list = []
                    x_list = []
                else:
                    s_list = check_list[0]
                    t_list = check_list[1]
                    x_list = check_list[2]

                fine_lable_s = torch.zeros(size=(total_data, data_shape[0], data_shape[1]))
                for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                    if model.name in ["MTAD_GAT"]:
                        x, h = model(d.to(device), None)
                    elif model.name in ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN",
                                        "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                                        "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD",
                                        "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
                        x = None
                        if is_search:
                            if check_list is None:
                                x, s, t, _, _, _ = model(d.to(device), t_adj, s_adj)
                                s_list.append(s)
                                t_list.append(t)
                                x_list.append(x)
                            else:
                                s = s_list[i]
                                t = t_list[i]
                                x = x_list[i]
                        else:
                            x, s, t, _, _, _ = model(d.to(device), t_adj, s_adj)
                            x = torch.clamp(x, min=0.0, max=1.0)
                        coff_s, s_m, _ = cacluateJaccard_optimized(t_adj=t_adj, test_adj=t, K=t_len,
                                                                   features=data_shape, type="s",
                                                                   up_threshold=up_threshould, down_threshold=0.0)
                        coff_t, t_m, _ = cacluateJaccard_optimized(t_adj=s_adj, test_adj=s, K=s_len,
                                                                       features=data_shape, type="t",
                                                                       up_threshold=up_threshould, down_threshold=0.0)
                        diff_metric = torch.abs(x - d.to(device)).view(s_len, t_len)
                        coff_rec = torch.mean(torch.mean(diff_metric)).cpu().detach().numpy()
                        rec_label = torch.zeros(size=data_shape)
                        max_idx = torch.argmax(diff_metric)
                        # Convert flat index to 2D coordinates (row, column)
                        rows, cols = s_len, t_len
                        row = max_idx // cols
                        col = max_idx % cols
                        rec_label[row][col] = 1

                        fine_lable[i] = torch.mul(s_m, t_m).to(torch.int32) | rec_label.to(torch.int32)  # Spatiotemporal coordinates
                        fine_lable_s[i] = s_m
                        coff_t = 1 - coff_t
                        coff_s = 1 - coff_s
                        jacced.append([2 * (coff_s * coff_t) / (coff_s + coff_t + 0.00001), coff_s,
                                       coff_t, coff_rec])

                    else:
                        x = model(d.to(device))
                    xs.append(x)
                jacced_ori = jacced
                jacced = np.asarray(jacced)
                jacced_means = np.mean(jacced, axis=0)
                jacced_stds = np.std(jacced, axis=0)
                three_sigma_thresholds_up = jacced_means + data_std * jacced_stds
                three_sigma_thresholds_down = jacced_means - data_std * jacced_stds

                pos_list = [np.argmin(np.asarray(jacced)[s_len:, 1]) + s_len,
                            np.argmin(np.asarray(jacced)[s_len:, 2]) + s_len]

                if model.name in ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN",
                                  "HAO_E_AHGSD", "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD",
                                  "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                                  "HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD"]:
                    for pos in pos_list:
                        x, t, s, _, _, _ = model(data[pos].to(device), t_adj, s_adj)
                        adj_list_t.append(t.cpu().detach().numpy())
                        adj_list_s.append(s.cpu().detach().numpy())
                xs = torch.stack(xs)
                y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)

                loss = l(xs.to(device), data.to(device))
                loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                ori_data = data[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                fine_lable = fine_lable[:, -1, :]
                try:
                    del xs
                    gc.collect()
                    torch.cuda.empty_cache()

                except:

                    print(traceback.print_exc())
                jacced = np.where(
                    (jacced > three_sigma_thresholds_up) | (jacced < three_sigma_thresholds_down),
                    1,
                    0
                )
                for i, (f, s, t, r) in enumerate(jacced):

                    if s == 0 and t == 0 and r == 0:
                        fine_lable[i] = 0
                    if s == 1 or t == 1 or r == 1:
                        index_sensor = torch.argmax(fine_lable_s[i][-1])
                        fine_lable[i][index_sensor] = 1
                return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), (
                adj_list_t, adj_list_s, jacced, fine_lable, jacced_ori, s_list, t_list,
                x_list), ori_data.cpu().detach().numpy()
            else:
                xs = []
                total_data = len(data)
                for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                    if model.name in ["MTAD_GAT"]:
                        x, h = model(d.to(device), None)
                    else:
                        x = model(d.to(device))
                    xs.append(x)
                xs = torch.stack(xs)
                y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                loss = l(xs.to(device), data.to(device))
                loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)

                return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), None, data.to(device)

    # Handle TranAD model
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = data
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = data_x.shape[1]
        l1s, l2s = [], []
        if training:
            total_data = len(data)
            for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                d = d.to(device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window.to(device), elem.to(device))
                l1 = l(z.to(device), elem.to(device)) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (
                        1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            total_data = len(data)
            outputs = []
            for i, d in tqdm(enumerate(data), total=total_data, desc=f'Epoch {epoch}'):
                d = d.reshape(1, w_size, d.shape[1])
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window.to(device), elem.to(device))
                if isinstance(z, tuple): z = z[1]
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs
            loss = l(outputs[:, -1, -1, :].to(device), data[:, -1, :].to(device))
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), None, data.to(device)
    else:
        y_pred = model(data.to(device))
        loss = l(y_pred.to(device), data.to(device))
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), None, data.to(device)



def get_finelabel(dataset, item_dataSet, batch_size, desc, label, model, search_spliteData, windows, features,
                  test_shape, accept_models):
    """
    Retrieves the best fine-grained labels and corresponding metrics for anomaly detection.

    Args:
        dataset (str): Name of the dataset (e.g., 'ASD', 'SMD').
        item_dataSet (str): Specific subset of the dataset.
        batch_size (int): Batch size used during training/testing.
        desc (str): Description or identifier for the experiment.
        label (torch.Tensor): Ground truth labels for the test data.
        model (str): Name of the model being evaluated.
        search_spliteData (list): List of data splitting ratios for validation/testing.
        windows (int): Number of windows used in the model.
        features (int): Number of features in the dataset.
        test_shape (int): Shape of the test data.
        accept_models (list): List of accepted models for evaluation.

    Returns:
        tuple: A tuple containing:
            - max_f1 (float): Maximum F1 score achieved.
            - max_fine_label (np.ndarray): Best fine-grained labels corresponding to the maximum F1 score.
            - max_correct_label (np.ndarray): Correct labels corresponding to the maximum F1 score.
            - label_fine_path_o (str): File path of the best fine-grained labels.
            - best_up (dict): Dictionary containing the best hyperparameters and thresholds.

    Functionality:
        1. Processes the ground truth labels to determine final binary labels (0 or 1).
        2. Iterates through different models and data splitting ratios to find the best-performing fine-grained labels.
        3. Loads precomputed fine-grained labels and evaluates their performance using F1 score.
        4. Updates the best results if a higher F1 score is achieved.
        5. Loads the best hyperparameters and thresholds from a JSON file if available.
        6. Returns the best F1 score, corresponding labels, file path, and hyperparameters.
    """
    # Initialize variables to track the best results
    max_f1 = -2  # Initialize to a very low value
    max_fine_label = None
    max_correct_label = None
    label_fine_path_o = None
    best_up_path = None
    current_best_up_path = None

    try:
        # Convert labels to numpy array and reshape if necessary
        labelY = label.cpu().detach().numpy()
        if labelY.shape[1] != features:
            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
        labelsFinal = (np.sum(labelY, axis=1) >= 1) + 0

    except:
        # Handle exceptions during label processing
        labelY = label.cpu().detach().numpy()
        if labelY.shape[1] != features:
            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
        labelsFinal = (np.sum(labelY, axis=1) >= 1) + 0
        labelsFinal = (np.sum(labelsFinal, axis=1) >= 1) + 0

    # Search for the best fine-grained labels across models and data splits
    if model not in accept_models:
        model = ["HAO_E", "HAO_P", "HAO_H"]
        for m in model:
            for s in search_spliteData:
                try:
                    # Construct paths for fine-grained labels and hyperparameters
                    label_fine_path = os.path.join(os.path.join("", "trainRecord/"),
                                                   "{}_{}/{}/{}/{}-{}/{}-{}".format(m, dataset,
                                                                                    item_dataSet,
                                                                                    batch_size, desc, windows,
                                                                                    m,
                                                                                    item_dataSet)) + str(
                        1) + "-" + f"windows_{windows}" + f"_epoch_{1}_{s}_Correctfinelabel.npy"
                    current_best_up_path = os.path.join(os.path.join("", "trainRecord/"),
                                                        "{}_{}/{}/{}/{}-{}/".format(m, dataset,
                                                                                    item_dataSet,
                                                                                    batch_size, desc, windows,
                                                                                    )) + f"model-{m}_{batch_size}_test_data_epoch_{1}_{s}.json"

                    # Load fine-grained labels
                    try:
                        fine_label = np.load(label_fine_path, allow_pickle=True)
                    except:
                        raise FileNotFoundError(f"Fine-grained label file not found: {label_fine_path}")

                    # Compute correct labels (at least one anomaly)
                    Correctfinelabel = (np.sum(fine_label, axis=1) >= 1).astype(int)

                    # Calculate weighted F1 score
                    weight = np.bincount(labelsFinal) / len(labelsFinal)
                    sample_weight = [weight[1] if item == 0 else weight[0] for item in labelsFinal]
                    if len(np.bincount(Correctfinelabel)) < 2:
                        f1 = 0.0
                    else:
                        f1 = f1_score(labelsFinal, Correctfinelabel, sample_weight=sample_weight, average="macro")

                    # Update best results if current F1 is higher
                    if f1 > max_f1:
                        max_f1 = f1
                        max_fine_label = fine_label
                        max_correct_label = Correctfinelabel
                        label_fine_path_o = label_fine_path
                        best_up_path = current_best_up_path
                except:
                    raise FileNotFoundError(f"Fine-grained label file not found: {label_fine_path}")
    else:
        # Search for the best fine-grained labels for the specified model
        for s in search_spliteData:
            try:
                # Construct paths for fine-grained labels and hyperparameters
                label_fine_path = os.path.join(os.path.join("", "trainRecord/"),
                                               "{}_{}/{}/{}/{}-{}/{}-{}".format(model, dataset,
                                                                                item_dataSet,
                                                                                batch_size, desc, windows,
                                                                                model,
                                                                                item_dataSet)) + str(
                    1) + "-" + f"windows_{windows}" + f"_epoch_{1}_{s}_Correctfinelabel.npy"
                current_best_up_path = os.path.join(os.path.join("", "trainRecord/"),
                                                    "{}_{}/{}/{}/{}-{}/".format(model, dataset,
                                                                                item_dataSet,
                                                                                batch_size, desc, windows,
                                                                                )) + f"model-{model}_{batch_size}_test_data_epoch_{1}_{s}.json"

                # Load fine-grained labels
                
                try:
                    fine_label = np.load(label_fine_path, allow_pickle=True)
                except:
                    raise FileNotFoundError(f"Fine-grained label file not found: {label_fine_path}")

                # Compute correct labels (at least one anomaly)
                Correctfinelabel = (np.sum(fine_label, axis=1) >= 1).astype(int)

                # Calculate weighted F1 score
                weight = np.bincount(labelsFinal) / len(labelsFinal)
                sample_weight = [weight[1] if item == 0 else weight[0] for item in labelsFinal]
                if len(np.bincount(Correctfinelabel)) < 2:
                    f1 = 0.0
                else:
                    f1 = f1_score(labelsFinal, Correctfinelabel, sample_weight=sample_weight, average="macro")

                # Update best results if current F1 is higher
                if f1 > max_f1:
                    max_f1 = f1
                    max_fine_label = fine_label
                    max_correct_label = Correctfinelabel
                    label_fine_path_o = label_fine_path
                    best_up_path = current_best_up_path
            except:
                raise FileNotFoundError(f"Fine-grained label file not found: {label_fine_path}")

    # Default hyperparameters
    best_up = {
        "s_best_f1": 0.0,
        "t_std_best_f1": 0.0,
        "s_std_best_f1": 0.0,
        "t_best_f1": 0.0,
        "s_up_threshold": 1.0,
        "s_down_threshold": 0.0,
        "t_up_threshould": 1.0,
        "t_down_threshold": 0.0,
        "s_std": 1.0,
        "t_std": 1.0,
    }

    # Load best hyperparameters from file if available
    try:
        with open(best_up_path, 'r', encoding='utf-8') as f:
            best_up = json.load(f)
    except:
        pass

    # Return the best results
    return max_f1, max_fine_label, max_correct_label, label_fine_path_o, best_up


def trainModel(model_name, dataset_name, epoch, windows, batch_size, item_dataSet, desc="",  total_epoch=0,is_corrected=False):
    """
    Trains and evaluates a model on a given dataset.

    Args:
        model_name (str): Name of the model to be trained.
        dataset_name (str): Name of the dataset (e.g., 'ASD', 'SMD').
        epoch (int): Starting epoch number.
        windows (int): Window size for sliding window conversion.
        batch_size (int): Batch size for training.
        item_dataSet (str): Specific subset of the dataset.
        desc (str, optional): Description or identifier for the experiment.
        total_epoch (int, optional): Total number of epochs to train.

    Functionality:
        1. Sets random seeds for reproducibility.
        2. Initializes data loaders for training and testing.
        3. Loads or creates the model, optimizer, and scheduler.
        4. Trains the model for the specified number of epochs.
        5. Evaluates the model using various anomaly detection methods.
        6. Saves results and metrics to JSON files.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    sklearn.utils.check_random_state(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    # Assign arguments to local variables
    batch_size = batch_size
    model = model_name
    dataset = dataset_name
    args.model = model
    args.dataset = dataset

    # Define data splitting ratios for validation/testing
    search_spliteData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # search_spliteData = [0.7]  # Use only 0.1 for simplicity
    is_corrected = is_corrected
    # Define POT quantile thresholds for different datasets
    pot_q = {
        "ASD": 0.01,
        "MSL": 0.001,
        "SMAP": 0.007,
        "SMD": 0.001,
        "SWaT": 0.001,
        "PSM": 0.001,
        "MSDS": 0.01,
        "WADI": 0.001,
        "synthetic": 0.001,
        "SCADA": 0.04,
        "PowerSystem": 0.01,
        "GAS": 0.001,
        "CICIDS": 0.18,
        "SKAB": 0.001,
        "NSL": 0.01,
        "CVES": 0.01,
        "WH": 0.03,
        "GECCO": 0.12,
        "NEGCCO": 0.1,
        "SWAN": 0.01,
    }

    # Initialize variables
    optimal_threshold = None
    data_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Train",
                             item_dataSet=item_dataSet)

    model, optimizer, scheduler, is_history = None, None, None, None
    features = data_loader.getDim()

    # Load or create the model
    if model_name not in ['MERLIN']:
        model, optimizer, scheduler, is_history = load_model(args.model, features, n_windows=windows,
                                                             desc=desc,
                                                             batch=batch_size, item_name=item_dataSet, epoch=epoch,
                                                             is_change_branch=data_loader.is_change_branch)
        model.to(device)

    # Create data loaders for training and testing
    trainStage = DataLoader(data_loader, batch_size=len(data_loader), shuffle=True, num_workers=0)
    test_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Test",
                             item_dataSet=item_dataSet)
    testStage = DataLoader(test_loader, batch_size=len(test_loader), shuffle=False)

    # Create output folder for saving results
    folder = f'trainRecord/{model_name}_{args.dataset}/{item_dataSet}/{batch_size}/{desc}-{windows}/'
    os.makedirs(folder, exist_ok=True)

    # Define accepted models for evaluation
    top_k = 1
    accept_models = ["HAO_E", "HAO_P", "HAO_H", "HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD",
                     "HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_H_HDNN", "HAO_H_T_HGCN",
                     "HAO_H_S_HGCN", "HAO_E_MSCD", "HAO_P_MSCD", "HAO_H_MSCD",
                     "HAO_H_AHGSD"]

    # Define paths for saving test results
    save_test_data = f'{folder}/model-{model_name}_{batch_size}_test_data_epoch_'
    save_test_pot_val_data = f'{folder}/model-{model_name}_{batch_size}_test_pot_val_data_epoch_'
    save_test_pot_test_data = f'{folder}/model-{model_name}_{batch_size}_test_pot_test_data_epoch_'
    save_test_tts_val_data = f'{folder}/model-{model_name}_{batch_size}_test_tts_val_data_epoch_'
    save_test_tts_test_data = f'{folder}/model-{model_name}_{batch_size}_test_tts_test_data_epoch_'
    save_test_sts_val_data = f'{folder}/model-{model_name}_{batch_size}_test_sts_val_data_epoch_'
    save_test_sts_test_data = f'{folder}/model-{model_name}_{batch_size}_test_sts_test_data_epoch_'
    save_test_nsts_val_data = f'{folder}/model-{model_name}_{batch_size}_test_nsts_val_data_epoch_'
    save_test_nsts_test_data = f'{folder}/model-{model_name}_{batch_size}_test_nsts_test_data_epoch_'

    # Initialize hyperparameters and counters
    patience_counter = 15
    test_std = np.linspace(0, 3, 20)
    timeOfPerDataTrian = None
    best_up = {
        "s_best_f1": 0.0,
        "t_std_best_f1": 0.0,
        "s_std_best_f1": 0.0,
        "t_best_f1": 0.0,
        "s_up_threshold": 1.0,
        "s_down_threshold": 0.0,
        "t_up_threshould": 1.0,
        "t_down_threshold": 0.0,
        "s_std": 1.0,
        "t_std": 1.0,
    }

    # Training phase
    if not args.test:
        start = time.time()
        data_struct = None
        Struct_M = {
            "Struct_rec": [],  # Predicted data
            "Struct_real": [],  # Real data
            "Struct_sdiff": [],  # Spatial structure difference
            "Struct_tdiff": [],  # Temporal structure difference
            "Struct_fdiff": [],  # Harmonic structure difference
            "Struct_anomaly": [],  # Fusion anomaly score
            "Struct_averageanomaly": [],  # MSE anomaly score
            "Struct_predict": [],  # Predicted labels
            "Struct_ture": [],  # True labels
        }

        # Loop over epochs
        for e in list(range(total_epoch - epoch + 1, total_epoch + 1)):
            # Handle MERLIN algorithm separately
            if model_name in ["MERLIN"]:
                
                train_loader, test_loader, labels = load_dataset(args.dataset, is_corrected)
                if is_corrected:
                    labels = torch.FloatTensor(labels).to(device).to(torch.float64)
                    _, _, _, path_fine, best_up = get_finelabel(dataset, item_dataSet, batch_size, desc,
                                                                labels, model_name, search_spliteData,
                                                                windows=windows, features=features,
                                                                test_shape=len(labels),
                                                                accept_models=accept_models)
                    labels = np.load(path_fine, allow_pickle=True)
                run_merlin(test_loader, labels, args.dataset, folder + f"{e}-{is_corrected}")
                continue

            # Clear GPU cache
            torch.cuda.empty_cache()

            # Define paths for saving adjacency matrices and training logs
            train_state_adj = [f'{folder}/{model_name}_t_adj.npy', f'{folder}/{model_name}_s_adj.npy',
                               f'{folder}/{model_name}_training_process_{batch_size}_{e}.json',
                               f'{folder}/{model_name}_t_adj.csv',
                               f'{folder}/{model_name}_s_adj.csv', f'{folder}/{model_name}_t_d_adj.csv',
                               f'{folder}/{model_name}_s_d_adj.csv']

            # Set model to training mode
            model.train()
            print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')

            # Training loop
            for train_data in trainStage:
                _, _, _, is_history = load_model(args.model, features, n_windows=windows, desc=desc, batch=batch_size,
                                                 item_name=item_dataSet, epoch=epoch,
                                                 is_change_branch=data_loader.is_change_branch)
                if is_history:
                    break
                else:
                    start_train_time = time.time()
                _, _ = backprop(e, model, train_data, features, optimizer, scheduler, training=True,
                                adj_save=train_state_adj, up_threshould=best_up["s_up_threshold"],
                                data_std=best_up["s_std"])
                if not is_history:
                    timeOfPerDataTrian = (time.time() - start_train_time) / len(train_data)
                save_model(model, optimizer, scheduler, windows, item_name=item_dataSet, desc=desc, batch=batch_size,
                           epoch=e)

            # Clean up memory
            try:
                del train_data
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
                print(traceback.print_exc())

            # Testing phase
            torch.zero_grad = True
            model.eval()

            with torch.no_grad():
                print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
                if is_corrected:
                    print("Anomaly Correction Mode")
                    for test, label, _ in testStage:
                        _, _, _, path_fine, _ = get_finelabel(dataset, item_dataSet, batch_size, desc,
                                                                    label, model_name, search_spliteData,
                                                                    windows=windows, features=features,
                                                                    test_shape=test.shape[0],
                                                                    accept_models=accept_models)
                        test_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows,
                                                 stage="Test",
                                                 item_dataSet=item_dataSet, path=path_fine)
                        testStage = DataLoader(test_loader, batch_size=len(test_loader), shuffle=False)

                # Evaluate the model
                for test, label, _ in testStage:
                    test_shape = test.shape[0]
                    try:
                        labelY = label.cpu().detach().numpy()
                        if labelY.shape[1] != features:
                            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
                        labelsFinal = (np.sum(labelY, axis=1) >= 1) + 0
                    except:
                        labelY = label.cpu().detach().numpy()
                        if labelY.shape[1] != features:
                            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
                        labelsFinal = (np.sum(labelY, axis=1) >= 1) + 0
                        labelsFinal = (np.sum(labelsFinal, axis=1) >= 1) + 0
                    lossT = None

                    # Calculate reconstruction differences for training data
                    print("Anomaly Distribution Mining Based on the POT Anomaly Detection Model:")
                    for train_data in trainStage:
                        lossT, _, adj, _ = backprop(0, model, train_data, features, optimizer,
                                                    scheduler,
                                                    training=False,
                                                    adj_save=train_state_adj, up_threshould=best_up["s_up_threshold"],
                                                    data_std=best_up["s_std"])
                        if model_name in accept_models:
                            lossT = adj[3].cpu().detach().numpy()

                    # Calculate reconstruction differences for test data
                    print("Anomaly Score Calculation for Test Samples:")
                    check_list = None
                    loss, y_preds, adj, ori_data = backprop(0, model, test, features, optimizer,
                                                            scheduler,
                                                            training=False,
                                                            adj_save=train_state_adj, is_search=True,
                                                            up_threshould=best_up["s_up_threshold"],
                                                            data_std=best_up["s_std"], check_list=check_list)

                    if model_name in accept_models:
                        check_list = (adj[5], adj[6], adj[7])

                    # Loop over data splitting ratios
                    for splite_data in search_spliteData:
                        if model_name in accept_models:
                            current_best_up_path = os.path.join(os.path.join("", "trainRecord/"),
                                                                "{}_{}/{}/{}/{}-{}/".format(model_name, dataset,
                                                                                            item_dataSet,
                                                                                            batch_size, desc, windows,
                                                                                            )) + f"model-{model_name}_{batch_size}_test_data_epoch_{1}_{splite_data}.json"
                            try:
                                with open(current_best_up_path, 'r', encoding='utf-8') as f:
                                    best_up = json.load(f)
                            except:
                                pass

                        # Split data into validation and test sets
                        labelY = label.cpu().detach().numpy()
                        if labelY.shape[1] != features:
                            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
                        neg_test_index = np.where(labelsFinal > 0)
                        pos_test_index = np.where(labelsFinal == 0)
                        samper_num_pos = int(len(pos_test_index[0]) * splite_data)
                        samper_num_neg = int(len(neg_test_index[0]) * splite_data)
                        all_index = set(list(pos_test_index[0])).union(set(list(neg_test_index[0])))
                        sampler_neg = np.random.choice(neg_test_index[0], size=samper_num_neg, replace=False)
                        sampler_pos = np.random.choice(pos_test_index[0], size=samper_num_pos, replace=False)
                        val_index = list(sampler_neg) + list(sampler_pos)
                        test_index = list(all_index - set(sampler_neg) - set(sampler_pos))

                        # Hyperparameter search phase
                        if model_name in accept_models:
                            best_up = {
                                "s_best_f1": 0.0,
                                "t_std_best_f1": 0.0,
                                "s_std_best_f1": 0.0,
                                "t_best_f1": 0.0,
                                "s_up_threshold": 1.0,
                                "s_down_threshold": 0.0,
                                "t_up_threshould": 1.0,
                                "t_down_threshold": 0.0,
                                "s_std": 1.0,
                                "t_std": 1.0,
                            }

                            # Coupling relationship upper bound search
                            search_count = 0
                            high = 1.0
                            low = 0.9
                            eps = 1e-8
                            estimated_iterations = int(np.log2((high - low) / eps)) + 1

                            print("Stage 1 of Hyperparameter Search for Natural Graph Sparsification: Based on Relation Parsing")
                            with tqdm(total=estimated_iterations) as pbar:
                                while high - low > eps:
                                    mid = (low + high) / 2

                                    loss, y_preds, adj, ori_data = backprop(0, model, test, features, optimizer,
                                                                            scheduler,
                                                                            training=False,
                                                                            adj_save=train_state_adj, up_threshould=mid,
                                                                            data_std=best_up["s_std"], is_search=True,
                                                                            check_list=check_list)

                                    val_data = np.concatenate([ori_data,
                                                               np.expand_dims(np.asarray(adj[2])[:, 1], axis=1),
                                                               np.expand_dims(np.asarray(adj[2])[:, 2], axis=1),
                                                               np.expand_dims(np.asarray(adj[2])[:, 3], axis=1)],
                                                              axis=1)[val_index]
                                    val_label = np.concatenate([labelY,
                                                                np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                                np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                                np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape)],
                                                               axis=1)[val_index]
                                    val_pred = np.concatenate([y_preds,
                                                               np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 1], axis=1).shape),
                                                               np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 2], axis=1).shape),
                                                               np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 3], axis=1).shape)],
                                                              axis=1)[val_index]

                                    s_val = [val_pred, val_data, val_label]
                                    test_scores, val_scores = get_full_err_scores_tra(s_val, s_val)
                                    result = get_best_performance_data_ori(val_scores, s_val[2].T, topk=top_k, average="macro")
                                    f1 = result["f1"]
                                    if f1 >= best_up["s_best_f1"]:
                                        search_count = 0
                                        best_up["s_best_f1"] = f1
                                        best_up["t_best_f1"] = f1
                                        best_up["s_up_threshold"] = mid
                                        best_up["t_up_threshold"] = mid
                                    else:
                                        search_count += 1

                                    best_up["search_count"] = search_count
                                    low, high = (mid, high) if f1 > 0.0 else (low, mid)

                                    if search_count > patience_counter:
                                        break

                                    pbar.update(1)

                            # Variance search
                            search_count = 0
                            loss, y_preds, adj, ori_data = backprop(0, model, test, features, optimizer,
                                                                    scheduler,
                                                                    training=False,
                                                                    adj_save=train_state_adj,
                                                                    up_threshould=best_up["s_up_threshold"],
                                                                    data_std=best_up["s_std"],
                                                                    is_search=True,
                                                                    check_list=check_list)

                            val_data = np.concatenate([ori_data,
                                                       np.expand_dims(np.asarray(adj[2])[:, 1], axis=1),
                                                       np.expand_dims(np.asarray(adj[2])[:, 2], axis=1),
                                                       np.expand_dims(np.asarray(adj[2])[:, 3], axis=1)],
                                                      axis=1)[val_index]
                            val_label = np.concatenate([labelY,
                                                        np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                        np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                        np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape)],
                                                       axis=1)[val_index]
                            val_pred = np.concatenate([y_preds,
                                                       np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 1], axis=1).shape),
                                                       np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 2], axis=1).shape),
                                                       np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 3], axis=1).shape)],
                                                      axis=1)[val_index]

                            s_val = [val_pred, val_data, val_label]
                            test_scores, val_scores = get_full_err_scores_tra(s_val, s_val)
                            result = get_best_performance_data_ori(val_scores, s_val[2].T, topk=top_k, average="macro")
                            best_up["s_best_f1"] = result["f1"]
                            best_up["t_best_f1"] = result["f1"]

                            best_up["s_std_best_f1"] = best_up["s_best_f1"]
                            best_up["t_std_best_f1"] = best_up["s_best_f1"]

                            print("Stage 2 of Hyperparameter Search for Natural Graph Sparsification: Based on Anomaly Distribution")
                            for mid in tqdm(test_std):
                                jacced = np.asarray(adj[4])
                                jacced_means = np.mean(jacced, axis=0)
                                jacced_stds = np.std(jacced, axis=0)
                                three_sigma_thresholds_up = jacced_means + mid * jacced_stds
                                three_sigma_thresholds_down = jacced_means - mid * jacced_stds
                                jacced = np.where((jacced > three_sigma_thresholds_up) | (jacced < three_sigma_thresholds_down), 1, 0)
                                val_data = np.concatenate([ori_data,
                                                           np.expand_dims(np.asarray(jacced)[:, 1], axis=1),
                                                           np.expand_dims(np.asarray(jacced)[:, 2], axis=1),
                                                           np.expand_dims(np.asarray(jacced)[:, 3], axis=1)],
                                                          axis=1)[val_index]
                                val_label = np.concatenate([labelY,
                                                            np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 0], axis=1).shape),
                                                            np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 0], axis=1).shape),
                                                            np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 0], axis=1).shape)],
                                                           axis=1)[val_index]
                                val_pred = np.concatenate([y_preds,
                                                           np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 1], axis=1).shape),
                                                           np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 2], axis=1).shape),
                                                           np.zeros(shape=np.expand_dims(np.asarray(jacced)[:, 3], axis=1).shape)],
                                                          axis=1)[val_index]

                                s_val = [val_pred, val_data, val_label]
                                test_scores, val_scores = get_full_err_scores_tra(s_val, s_val)
                                result = get_best_performance_data_ori(val_scores, s_val[2].T, topk=top_k, average="macro")
                                f1 = result["f1"]
                                if f1 >= best_up["s_std_best_f1"]:
                                    search_count = 0
                                    best_up["s_std_best_f1"] = f1
                                    best_up["t_std_best_f1"] = f1
                                    best_up["s_std"] = mid
                                    best_up["t_std"] = mid
                                else:
                                    search_count += 1
                                best_up["search_count"] = search_count

                        # Test data evaluation phase
                        if model_name in accept_models:
                            loss, y_preds, adj, ori_data = backprop(0, model, test, features, optimizer,
                                                                    scheduler,
                                                                    training=False,
                                                                    adj_save=train_state_adj,
                                                                    is_search=True,
                                                                    up_threshould=best_up["s_up_threshold"],
                                                                    data_std=best_up["s_std"],
                                                                    check_list=check_list)

                        # Save results
                        test_shape = test.shape[0]
                        save_data = dict()
                        if model_name in accept_models:
                            loss = adj[3].cpu().detach().numpy()
                            if splite_data == 0.1:
                                save_data["real"] = ori_data.tolist()
                                save_data["pred"] = y_preds.tolist()
                                save_data["label"] = labelsFinal.tolist()

                            save_data["f_diff"] = np.asarray(adj[2])[:, 0].tolist()
                            save_data["s_diff"] = np.asarray(adj[2])[:, 1].tolist()
                            save_data["t_diff"] = np.asarray(adj[2])[:, 2].tolist()
                            save_data["r_diff"] = np.asarray(adj[2])[:, 3].tolist()

                        save_data["s_best_f1"] = best_up["s_best_f1"]
                        save_data["t_std_best_f1"] = best_up["t_std_best_f1"]
                        save_data["s_std_best_f1"] = best_up["s_std_best_f1"]
                        save_data["t_best_f1"] = best_up["t_best_f1"]
                        save_data["s_up_threshold"] = best_up["s_up_threshold"]
                        save_data["s_down_threshold"] = best_up["s_down_threshold"]
                        save_data["t_up_threshould"] = best_up["t_up_threshould"]
                        save_data["t_down_threshold"] = best_up["t_down_threshold"]
                        save_data["s_std"] = best_up["s_std"]
                        save_data["t_std"] = best_up["t_std"]
                        ori_data = tensor_to_numpy(ori_data)
                        if ori_data.shape[1] != features:
                            ori_data = ori_data.reshape(test_shape, -1, features)[:, -1, :]
                        y_preds = tensor_to_numpy(y_preds)
                        if y_preds.shape[1] != features:
                            y_preds = y_preds.reshape(test_shape, -1, features)[:, -1, :]
                        save_data["mse"] = np.mean((ori_data - y_preds) ** 2)
                        save_data["model_total_params"] = sum(p.numel() for p in model.parameters())
                        save_data["model_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        if not is_history:
                            save_data["train_time"] = timeOfPerDataTrian

                        path = f'{save_test_data}{e}_{splite_data}.json'
                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=4)

                        # POT method detection
                        lossFinal = np.mean(loss, axis=1)
                        lossTfinal = np.mean(lossT, axis=1)

                        val_data = lossFinal[val_index]
                        val_label = labelsFinal[val_index]
                        test_data = lossFinal[test_index]
                        test_label = labelsFinal[test_index]

                        result, pred_pot = pot_eval(lossTfinal, val_data, val_label, q=pot_q[dataset_name])
                        print("**********Traditional POT Threshold Detection Algorithm**********************************************************")
                        print("POT macro val results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                            result["f1"],
                            result["recall"],
                            result["precision"], result["AP"], result["ROC/AUC"],
                            result["threshold"]))
                        save_data = dict()
                        save_data["p_latency"] = result["p_latency"]

                        l = label.view(loss.shape[0], -1, loss.shape[1])[:, -1, :].cpu().detach().numpy()[val_index]
                        res = hit_att(loss[val_index], l)

                        save_data["Hit@500%"] = res["Hit@500%"]
                        save_data["Hit@600%"] = res["Hit@600%"]

                        res = ndcg(loss[val_index], l)

                        save_data["NDCG@500%"] = res["NDCG@500%"]
                        save_data["NDCG@600%"] = res["NDCG@600%"]
                        save_data["F1"] = result["f1"]
                        save_data["recall"] = result["recall"]
                        save_data["precision"] = result["precision"]
                        save_data["AP"] = result["AP"]
                        save_data["AUC"] = result["ROC/AUC"]
                        save_data["threshold"] = result["threshold"]
                        save_data["ACC"] = result["ACC"]
                        save_data["anomaly_scores"] = val_data.tolist()
                        save_data["gt_label"] = val_label.tolist()
                        save_data["pred_label"] = pred_pot.tolist()
                        path = f'{save_test_pot_val_data}{e}_{splite_data}.json'

                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=4)

                        result = testThreshouldPerfermance_pot(test_data, test_label, result["threshold"], "macro")
                        print("POT macro test results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f}".format(
                            result["f1"],
                            result["recall"],
                            result["precision"], result["AP"], result["roc_auc"]))
                        print("********************************************************************")
                        save_data = dict()
                        save_data["p_latency"] = result["p_latency"]
                        l = label.view(loss.shape[0], -1, loss.shape[1])[:, -1, :].cpu().detach().numpy()[test_index]
                        res = hit_att(loss[test_index], l)

                        save_data["Hit@500%"] = res["Hit@500%"]
                        save_data["Hit@600%"] = res["Hit@600%"]
                        res = ndcg(loss[test_index], l)

                        save_data["NDCG@500%"] = res["NDCG@500%"]
                        save_data["NDCG@600%"] = res["NDCG@600%"]

                        save_data["F1"] = result["f1"]
                        save_data["recall"] = result["recall"]
                        save_data["precision"] = result["precision"]
                        save_data["accuracy"] = result["accuracy"]
                        save_data["AUC"] = result["roc_auc"]
                        save_data["AP"] = result["AP"]
                        save_data["threshold"] = result["threshold"]
                        save_data["anomaly_scores"] = test_data.tolist()
                        save_data["gt_label"] = test_label.tolist()
                        save_data["pred_label"] = result["pred"].tolist()
                        path = f'{save_test_pot_test_data}{e}_{splite_data}.json'

                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=4)

                        # Traditional threshold search algorithm
                        labelY = label.cpu().detach().numpy()
                        if labelY.shape[1] != features:
                            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]

                        val_data = ori_data[val_index]
                        val_label = labelY[val_index]
                        val_pred = y_preds[val_index]

                        test_data = ori_data[test_index]
                        test_label = labelY[test_index]
                        test_pred = y_preds[test_index]

                        s_test = [test_pred, test_data, test_label]
                        s_val = [val_pred, val_data, val_label]
                        test_scores, val_scores = get_full_err_scores_ori(s_test, s_val)

                        result = get_best_performance_data_tra(val_scores, s_val[2].T, topk=top_k, average="macro")
                        print("**********Traditional Threshold Search Algorithm*********************************")
                        print("TTS macro val results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                            result["f1"], result["recall"], result["precision"], result["AP"], result["roc_auc"],
                            result["threshold"]))
                        save_data = dict()
                        save_data["F1"] = result["f1"]
                        save_data["recall"] = result["recall"]
                        save_data["precision"] = result["precision"]
                        save_data["accuracy"] = result["accuracy"]
                        save_data["AUC"] = result["roc_auc"]
                        save_data["AP"] = result["AP"]
                        save_data["threshold"] = result["threshold"]
                        save_data["anomaly_scores"] = result["scores_search"]
                        save_data["gt_label"] = result["gt_labels"]
                        save_data["pred_label"] = result["y_pred"]
                        path = f'{save_test_tts_val_data}{e}_{splite_data}.json'
                        Struct_M["gt_label_val_index"] = val_index
                        Struct_M["pred_label_test_index"] = test_index
                        Struct_M["Struct_val_gt"] = result["gt_labels"]
                        Struct_M["Struct_val_pred"] = result["y_pred"]
                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=4)

                        result = testThreshouldPerfermance_Trandation(test_scores, s_test[2].T,
                                                                      thresold=result["threshold"], topk=top_k,
                                                                      average="macro")
                        print("TTS macro test results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                            result["f1"], result["recall"], result["precision"], result["AP"], result["roc_auc"],
                            result["threshold"]))
                        print("********************************************************************")
                        save_data = dict()
                        save_data["F1"] = result["f1"]
                        save_data["recall"] = result["recall"]
                        save_data["precision"] = result["precision"]
                        save_data["accuracy"] = result["accuracy"]
                        save_data["AUC"] = result["roc_auc"]
                        save_data["AP"] = result["AP"]
                        save_data["threshold"] = result["threshold"]
                        save_data["anomaly_scores"] = result["scores_search"]
                        save_data["gt_label"] = result["gt_labels"]
                        save_data["pred_label"] = result["y_pred"]

                        path = f'{save_test_tts_test_data}{e}_{splite_data}.json'

                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=4)

                        # Structure-aware threshold search algorithm
                        if model_name in accept_models:
                            labelY = label.cpu().detach().numpy()
                            if labelY.shape[1] != features:
                                labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
                            labelY = np.concatenate([labelY,
                                                     np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                     np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                                     np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape)],
                                                    axis=1)

                            Train_pre = np.concatenate([ori_data,
                                                        np.expand_dims(np.asarray(adj[2])[:, 1], axis=1),
                                                        np.expand_dims(np.asarray(adj[2])[:, 2], axis=1),
                                                        np.expand_dims(np.asarray(adj[2])[:, 3], axis=1)],
                                                       axis=1)
                            Y_preds = np.concatenate([y_preds,
                                                      np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 1], axis=1).shape),
                                                      np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 2], axis=1).shape),
                                                      np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 3], axis=1).shape)],
                                                     axis=1)

                            val_data = Train_pre[val_index]
                            val_label = labelY[val_index]
                            val_pred = Y_preds[val_index]

                            test_data = Train_pre[test_index]
                            test_label = labelY[test_index]
                            test_pred = Y_preds[test_index]

                            s_test = [test_pred, test_data, test_label]
                            s_val = [val_pred, val_data, val_label]

                            test_scores, val_scores = get_full_err_scores_tra(s_test, s_val)
                            result = get_best_performance_data_ori(val_scores, s_val[2].T, topk=top_k, average="macro")
                            print("**********Proposed Structure-Aware Threshold Search Algorithm***************************************")
                            print("STS macro val results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                                result["f1"], result["recall"], result["precision"], result["AP"],
                                result["roc_auc"],
                                result["threshold"]))
                            save_data = dict()
                            save_data["F1"] = result["f1"]
                            save_data["recall"] = result["recall"]
                            save_data["precision"] = result["precision"]
                            save_data["accuracy"] = result["accuracy"]
                            save_data["AUC"] = result["roc_auc"]
                            save_data["AP"] = result["AP"]
                            save_data["threshold"] = result["threshold"]
                            save_data["anomaly_scores"] = result["scores_search"]
                            save_data["gt_label"] = result["gt_labels"]
                            save_data["pred_label"] = result["y_pred"]
                            path = f'{save_test_sts_val_data}{e}_{splite_data}.json'

                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=4)
                            result = testThreshouldPerfermance_Trandation(test_scores, s_test[2].T,
                                                                          thresold=result["threshold"], topk=top_k,
                                                                          average="macro")

                            print("STS macro test results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                                result["f1"], result["recall"], result["precision"], result["AP"],
                                result["roc_auc"],
                                result["threshold"]))
                            save_data = dict()
                            save_data["F1"] = result["f1"]
                            save_data["recall"] = result["recall"]
                            save_data["precision"] = result["precision"]
                            save_data["accuracy"] = result["accuracy"]
                            save_data["AUC"] = result["roc_auc"]
                            save_data["AP"] = result["AP"]
                            save_data["threshold"] = result["threshold"]
                            save_data["anomaly_scores"] = result["scores_search"]
                            save_data["gt_label"] = result["gt_labels"]
                            save_data["pred_label"] = result["y_pred"]
                            path = f'{save_test_sts_test_data}{e}_{splite_data}.json'

                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=4)

                            # Non-structure-aware threshold search algorithm
                            labelY = label.cpu().detach().numpy()
                            if labelY.shape[1] != features:
                                labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]

                            val_data = ori_data[val_index]
                            val_label = labelY[val_index]
                            val_pred = y_preds[val_index]

                            test_data = ori_data[test_index]
                            test_label = labelY[test_index]
                            test_pred = y_preds[test_index]

                            s_test = [test_pred, test_data, test_label]
                            s_val = [val_pred, val_data, val_label]
                            test_scores, val_scores = get_full_err_scores_tra(s_test, s_val)
                            result = get_best_performance_data_ori(val_scores, s_val[2].T, topk=top_k, average="macro")
                            print("**********Proposed Non-Structure-Aware Threshold Search Algorithm*********************************")
                            print("NSTS macro val results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                                result["f1"], result["recall"], result["precision"], result["AP"],
                                result["roc_auc"], result["threshold"]))
                            save_data = dict()
                            save_data["F1"] = result["f1"]
                            save_data["recall"] = result["recall"]
                            save_data["precision"] = result["precision"]
                            save_data["accuracy"] = result["accuracy"]
                            save_data["AUC"] = result["roc_auc"]
                            save_data["AP"] = result["AP"]
                            save_data["threshold"] = result["threshold"]
                            save_data["anomaly_scores"] = result["scores_search"]
                            save_data["gt_label"] = result["gt_labels"]
                            save_data["pred_label"] = result["y_pred"]
                            path = f'{save_test_nsts_val_data}{e}_{splite_data}.json'

                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=4)

                            result = testThreshouldPerfermance_Trandation(test_scores, s_test[2].T,
                                                                          thresold=result["threshold"], topk=top_k,
                                                                          average="macro")
                            print("NSTS macro test results: F1:{:.6f},Rec:{:.6f},Pre:{:.6f},AP:{:.6f},AUC:{:.6f},Threshold:{:.6f}".format(
                                result["f1"], result["recall"], result["precision"], result["AP"],
                                result["roc_auc"], result["threshold"]))
                            save_data = dict()
                            save_data["F1"] = result["f1"]
                            save_data["recall"] = result["recall"]
                            save_data["precision"] = result["precision"]
                            save_data["accuracy"] = result["accuracy"]
                            save_data["AUC"] = result["roc_auc"]
                            save_data["AP"] = result["AP"]
                            save_data["threshold"] = result["threshold"]
                            save_data["anomaly_scores"] = result["scores_search"]
                            save_data["gt_label"] = result["gt_labels"]
                            save_data["pred_label"] = result["y_pred"]
                            path = f'{save_test_nsts_test_data}{e}_{splite_data}.json'

                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=4)
                        file_list = os.path.join(os.path.join("", "trainRecord/"),
                                                 "{}_{}/{}/{}/{}-{}/{}-{}".format(model_name, dataset, item_dataSet,
                                                                                  batch_size, desc, windows,
                                                                                  model_name, item_dataSet)) + str(
                            e) + "-" + f"windows_{windows}" + f"_epoch_{e}_{splite_data}"

                        if model_name in accept_models:

                            labelY = label.cpu().detach().numpy()
                            if labelY.shape[1] != features:
                                labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]

                            labelY = np.concatenate(
                                [labelY,
                                 # np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                 ],
                                axis=1)

                            # 扩充结构差异
                            Train_pre = np.concatenate([ori_data,
                                                        # np.expand_dims(np.asarray(adj[2])[:, 0], axis=1),
                                                        np.expand_dims(np.asarray(adj[2])[:, 1], axis=1),
                                                        np.expand_dims(np.asarray(adj[2])[:, 2], axis=1),
                                                        np.expand_dims(np.asarray(adj[2])[:, 3], axis=1),
                                                        ],
                                                       axis=1)
                            Y_preds = np.concatenate(
                                [y_preds,
                                 # np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 1], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 2], axis=1).shape),
                                 np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 3], axis=1).shape),
                                 ],
                                axis=1)

                            Struct_M["Struct_rec"] = y_preds
                            Struct_M["Struct_real"] = ori_data
                            Struct_M["Struct_sdiff"] = np.asarray(adj[2])[:, 1]
                            Struct_M["Struct_tdiff"] = np.asarray(adj[2])[:, 2]
                            Struct_M["Struct_fdiff"] = np.asarray(adj[2])[:, 0]
                            Struct_M["Struct_rdiff"] = np.asarray(adj[2])[:, 3]
                            s_all = [Train_pre, Y_preds, labelY]
                            test_scores_s, normal_scores_s = get_full_err_scores(s_all, s_all)
                            info = get_best_performance_data_ori(test_scores_s, s_all[2].T, topk=top_k)
                            Struct_M["Struct_anomaly"] = info["scores_search"]
                            Struct_M["fine_pred_label"] = adj[3]
                            Struct_M["y_pred"] = info["y_pred"]
                            Struct_M["Struct_averageanomaly"] = info["scores_diff"]
                            Struct_M["Struct_predict"] = info["y_pred"]
                            Struct_M["Struct_ture"] = info["gt_labels"]
                            Struct_M["scores_search"] = info["scores_search"]
                            Struct_M["Anomaly"] = np.mean(loss, axis=1)
                            Struct_M["Anomaly_max"] = np.max(np.abs(
                                np.subtract(np.array(y_preds).astype(np.float64),
                                            np.array(ori_data).astype(np.float64))), axis=1)
                            # print(best_up)
                            plotPried(true=Struct_M["Struct_real"],
                                      predict=Struct_M["Struct_rec"],
                                      labels=labelY,
                                      p_true_labels=labelsFinal,
                                      threshould=labelsFinal,
                                      asd=[],
                                      p_labels=[],
                                      wholeData=ori_data,
                                      # f1=optimal_metrics["f1"],
                                      stand=optimal_threshold,
                                      Train_name=dataset_name,
                                      item_name=item_dataSet,
                                      pot_f1=0.0,
                                      # s_f1=search_res["f1"],
                                      rec_loss=np.expand_dims(adj[2], axis=1),
                                      t_adj=train_state_adj[0],
                                      s_adj=train_state_adj[1],
                                      t_adj_t=adj[0],
                                      s_adj_t=adj[1],
                                      rec_loss_h=loss,
                                      Struct_mess=Struct_M,
                                      best_up=best_up,
                                      dataname=file_list,
                                      splite_data=splite_data)
                        else:
                            labelY = label.cpu().detach().numpy()
                            if labelY.shape[1] != features:
                                labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]

                            labelY = np.concatenate([labelY, ], axis=1)

                            # 扩充结构差异
                            Train_pre = np.concatenate([tensor_to_numpy(ori_data), ], axis=1)
                            Y_preds = np.concatenate(
                                [tensor_to_numpy(y_preds), ],
                                axis=1)

                            Struct_M["Struct_rec"] = y_preds
                            Struct_M["Struct_real"] = ori_data
                            # Struct_M["Struct_sdiff"] = np.asarray(adj[2])[:, 1]
                            # Struct_M["Struct_tdiff"] = np.asarray(adj[2])[:, 2]
                            # Struct_M["Struct_fdiff"] = np.asarray(adj[2])[:, 0]
                            s_all = [Train_pre, Y_preds, labelY]
                            test_scores_s, normal_scores_s = get_full_err_scores(s_all, s_all)
                            info = get_best_performance_data_ori(test_scores_s, s_all[2].T, topk=top_k)
                            Struct_M["Struct_anomaly"] = info["scores_search"]
                            # Struct_M["fine_pred_label"] = adj[3]
                            Struct_M["y_pred"] = info["y_pred"]
                            # Struct_M["Struct_averageanomaly"] = info["scores_diff"]
                            Struct_M["Struct_predict"] = info["y_pred"]
                            Struct_M["Struct_ture"] = info["gt_labels"]
                            Struct_M["scores_search"] = info["scores_search"]
                            Struct_M["Anomaly"] = np.mean(loss, axis=1)
                            Struct_M["Anomaly_max"] = np.max(np.abs(
                                np.subtract(np.array(tensor_to_numpy(y_preds)).astype(np.float64),
                                            np.array(tensor_to_numpy(ori_data)).astype(np.float64))), axis=1)
                            plotPried(true=np.array(tensor_to_numpy(Struct_M["Struct_real"])).astype(np.float64),
                                      predict=np.array(tensor_to_numpy(Struct_M["Struct_rec"])).astype(np.float64),
                                      labels=labelY,
                                      p_true_labels=labelsFinal,
                                      threshould=labelsFinal,
                                      asd=[],
                                      p_labels=[],
                                      wholeData=ori_data,
                                      # f1=optimal_metrics["f1"],
                                      stand=optimal_threshold,
                                      Train_name=dataset_name,
                                      item_name=item_dataSet,
                                      pot_f1=0.0,
                                      # s_f1=search_res["f1"],
                                      rec_loss=None,
                                      t_adj=None,
                                      s_adj=None,
                                      t_adj_t=None,
                                      s_adj_t=None,
                                      rec_loss_h=loss,
                                      Struct_mess=Struct_M,
                                      best_up=best_up,
                                      dataname=file_list,
                                      splite_data=splite_data)

    # Clean up memory
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        traceback.print_stack()

    


def getDataSetList(dataSet):
    """
    Retrieves a list of dataset item names for a given dataset.

    Args:
        dataSet (str): Name of the dataset (e.g., 'ASD', 'MSL', 'SMAP', 'SMD').

    Returns:
        list: A list of dataset item names (e.g., ['omi-1_', 'machine-3-7_']).

    Functionality:
        1. Constructs the folder path for the specified dataset.
        2. Checks if the folder exists; raises an exception if not found.
        3. Uses glob to find all files ending with '_train.npy' in the folder.
        4. Based on the dataset name, returns predefined item names or extracts them from filenames:
           - For specific datasets (e.g., 'ASD', 'MSL'), predefined item names are returned.
           - For other datasets, item names are extracted from the filenames by removing the folder path and suffix.
    """
    req = []
    folder = os.path.join(output_folder, dataSet)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')

    files = glob.glob(os.path.join(folder, "*train.npy"))
    if dataSet == "ASD":
        req = ["omi-1_", ]
    elif dataSet == "MSL":
        req = ["C-2_"]
    elif dataSet == "SMAP":
        req = ["A-4_"]
    elif dataSet == "SMD":
        req = ["machine-3-7_"]
    else:
        for file in files:
            req.append(file[len(folder) + 1:len(file) - len("_train.npy") + 1])
    return req


def find_files_with_name(folder_path, name_part):
    """
    Finds all files in a given folder that contain a specific substring in their names.

    Args:
        folder_path (str): Path to the folder where files are located.
        name_part (str): Substring to search for within file names.

    Returns:
        list: A list of file names that contain the specified substring.

    Functionality:
        1. Lists all items (files and directories) in the specified folder.
        2. Filters the list to include only files that contain the given substring in their names.
        3. Returns the filtered list of file names.
    """
    all_items = os.listdir(folder_path)
    
    filtered_files = [file for file in all_items if name_part in file]
    return filtered_files


def getTrainHistory(model, epoch_mod, dataset, batch=None, window="", desc="",batch_size=32):
    """
    Determines the remaining training epochs needed for each dataset item based on existing training records.

    Args:
        model (str): Name of the model being evaluated.
        epoch_mod (int): Total number of epochs planned for training.
        dataset (str): Name of the dataset (e.g., 'ASD', 'SMD').
        batch (int, optional): Batch size used during training.
        window (str, optional): Window size or identifier for the model.
        desc (str, optional): Description or identifier for the experiment.

    Returns:
        tuple: A tuple containing:
            - need_train_epoch (list): List of remaining epochs needed for each dataset item.
            - need_train (list): List of dataset items requiring training.

    Functionality:
        1. Constructs the base path for training records.
        2. Retrieves the list of dataset items that need training.
        3. Initializes a list to track the remaining epochs for each dataset item.
        4. For each dataset item:
           - Searches for existing training record folders.
           - Identifies relevant files based on naming conventions.
           - Calculates the difference between planned epochs and completed epochs.
           - Updates the remaining epochs accordingly.
        5. Returns the lists of remaining epochs and dataset items requiring training.
    """
    history_path = os.getcwd() + "/trainRecord"
    if not os.path.exists(history_path):
        raise Exception('Processed Data not found.')
    need_train = getDataSetList(dataSet=dataset)
    need_train_epoch = [epoch_mod for item in need_train]
    for index, n in enumerate(need_train):
        file_list = glob.glob(os.path.join(history_path, "{}_{}/{}/".format(model, dataset, n)))
        for item in file_list:
            item_path = item + "/{}/{}-{}/".format(batch, desc, window)
            files = None
            try:
                files = find_files_with_name(item_path, f"_{batch_size}_test_data_epoch_")
            except:
                files = []
            p = epoch_mod - len(files) // 9
            if p > 0:
                need_train_epoch[index] = p
                
            else:
                need_train_epoch[index] = 0
                

    return need_train_epoch, need_train


if __name__ == '__main__':
    desc = "Demo"
    batch_size = [256]
    Dataset = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT', 'PSM', 'MSDS', 'synthetic', "SCADA", "PowerSystem", "WADI", "GAS", "CICIDS", "SKAB", "SWAN", "NEGCCO"]
    Dataset= ["MSL"]
    
    # Ablation experiments
    models_e = ["HAO_E_HDNN", "HAO_E_T_HGCN", "HAO_E_S_HGCN", "HAO_E_AHGSD", "HAO_E_MSCD"]
    models_h = ["HAO_H_HDNN", "HAO_H_T_HGCN", "HAO_H_S_HGCN", "HAO_H_AHGSD", "HAO_H_MSCD"]
    models_p = ["HAO_P_HDNN", "HAO_P_T_HGCN", "HAO_P_S_HGCN", "HAO_P_AHGSD", "HAO_P_MSCD"]
    compare = ["MERLIN","OmniAnomaly", "MAD_GAN", "MSCRED", "MTAD_GAT", "GDN", "CAE_M", "TranAD", "STADN", "GRN", "DGINet","TS_GAT"]
    models = ["HAO_E", "HAO_P", "HAO_H"] + compare + models_e + models_h + models_p
    models = ["HAO_E"]
    epoch = 1
    WindowSize = [15]
    is_corrected = True
    # Comparative experiment testing
    for b in batch_size:
        for m in models:
            for d in Dataset:
                for window in WindowSize:
                    need_train_epoch, data_list = getTrainHistory(model=m, epoch_mod=epoch, dataset=d, batch=b,
                                                                  window=window, desc=desc,batch_size=b)
                    for e, item_dataSet in enumerate(data_list):
                        try:
                            res = trainModel(model_name=m, dataset_name=d, epoch=need_train_epoch[e], windows=window,
                                             batch_size=b, total_epoch=epoch,
                                             item_dataSet=item_dataSet, desc=desc,is_corrected=is_corrected)
                        except Exception as e:
                            print(traceback.print_exc())
