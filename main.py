import gc
import glob
import json
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.evaluate import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.NoPOT import *
import scipy.io as sio
import src.models
device = "cuda:0"
import warnings
import traceback
debug = False
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True


class dataLoader(Dataset):
    
    def __init__(self, DataName="", modelName="", convertWindow="", stage="Train", item_dataSet=""):
        """
        Initialize the dataset loader.

        Parameters:
        - DataName (str): Name of the dataset. Options include 'ASD', 'SMD', 'SMAP', 'MSL'.
        - modelName (str): Model name used to determine specific data processing logic.
        - convertWindow (int or str): Window size for sequence conversion. If not in `exception_models`, it will be determined automatically.
        - stage (str): Stage of usage, either 'Train' or others (likely 'Test').
        - item_dataSet (str): Identifier for the specific dataset item, useful for datasets containing multiple subsets.
        """
        self.DataName = DataName
        self.modelName = modelName
        self.convertWindow = convertWindow
        self.stage = stage    
        # Models that require special handling for window size
        self.limitModel = ['HAO_E', 'HAO_P', 'HAO_H']  
        # Stack models list (currently empty)
        self.limitStackModel = []  
        exception_models = self.limitModel  # Models requiring custom window size
        folder = os.path.join(output_folder, self.DataName)

        # Check if processed data exists
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')

        self.loader = []

        # Load train, test, and label files
        for file in ['train', 'test', 'labels']:
            if DataName == 'ASD': 
                file = item_dataSet + file
            if DataName == 'SMD': 
                file = item_dataSet + file
            if DataName == 'SMAP': 
                file = item_dataSet + file
            if DataName == 'MSL': 
                file = item_dataSet + file

            # Load numpy arrays and slice from index 20 onward
            self.loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True)[20:])

        # Convert labels to binary format (1 if sum >= 1, else 0)
        self.labes_ture = torch.tensor((np.sum(self.loader[2], axis=1) >= 1) + 0)
        
        # Count occurrences of each class
        self.labels_counts = torch.bincount(self.labes_ture)
        
        # Compute inverse class weights
        self.class_weights = 1.0 / self.labels_counts.float()
        
        # Assign sample weights based on class weights
        self.sample_weights = self.class_weights[torch.tensor(self.labes_ture)]

        # Set window size depending on model
        if self.modelName in exception_models:
            self.convertWindow = convertWindow
        else:
            self.convertWindow = self.getDim()  # Automatically determine window size

        # Preprocess data if model is in limited models
        if self.modelName in self.limitModel:
            self.trainData = self.convertWindowPro(0)  # Process training data
            self.testData = self.convertWindowPro(1)    # Process test data
            self.labelData = self.convertWindowPro(2)   # Process label data


    def getDim(self):
        """
        Get the dimensionality of the data (number of features).

        Returns:
        - int: Number of features in the dataset.
        """
        return self.loader[0].shape[1]

    def convertWindowPro(self, index):
        """
        Process the data by creating sliding windows for models that require windowing.

        For each data point, if the current index is beyond the window size, extract the window.
        Otherwise, pad the beginning with repeated initial data points to maintain window size.

        Parameters:
        - index (int): Index of the dataset (0 for train, 1 for test, etc.).

        Returns:
        - torch.Tensor: A tensor containing processed windows of data.
        """
        if self.modelName in self.limitModel:
            windows = []
            data = torch.Tensor(self.loader[index])
            for i, g in enumerate(self.loader[index]):
                if i >= self.convertWindow:
                    w = data[i - self.convertWindow:i]
                else:
                    # Pad the start with repeated initial data points
                    pad = data[0].repeat(self.convertWindow - i, 1)
                    w = torch.cat([pad, data[0:i]])
                if self.modelName in self.limitStackModel:
                    windows.append(w)
                else:
                    # Flatten the window if not using stacking
                    windows.append(w.contiguous().view(-1))
            return torch.stack(windows)

    def __len__(self):
        """
        Returns the length of the dataset based on the stage (Train or Test)
        and whether the model requires special data handling ([limitModel].

        If the model is in `limitModel`, it uses processed train/test data lengths;
        otherwise, it uses the raw loader's data length.

        Returns:
            int: Length of the dataset for the current stage.
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
        Retrieves a single data sample based on the index and current stage (Train or Test).

        If in training mode:
            - Returns processed training data if the model is in [limitModel].
            - Otherwise, returns raw loader training data.

        If in testing mode:
            - Returns processed test data along with corresponding labels if the model is in [limitModel].
            - Otherwise, returns raw loader test data and its corresponding labels.

        Args:
            item (int): Index of the data sample to retrieve.

        Returns:
            torch.Tensor: Data sample for training/testing.
            torch.Tensor (optional): Corresponding label for the test data.
        """
        if self.stage == "Train":
            if self.modelName in self.limitModel:
                return torch.FloatTensor(self.trainData[item]).to(device).to(torch.float64)
            else:
                return torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)
        else:
            if self.modelName in self.limitModel:
                return torch.FloatTensor(self.testData[item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.labelData[item]).to(device).to(torch.float64)
            else:
                return torch.FloatTensor(self.loader[1][item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.loader[2][item]).to(device).to(torch.float64)



def save_model(model, optimizer, scheduler, n_windows, item_name, batch=None, desc=""):
    """
    Saves the model and its associated optimizer and scheduler states to a file.

    Parameters:
    - model (torch.nn.Module): The neural network model to be saved.
    - optimizer (torch.optim.Optimizer): The optimizer used during training.
    - scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
    - n_windows (int): Size of the window used in data processing or model configuration.
    - item_name (str): Identifier for the specific dataset item being processed.
    - batch (optional, int or str): Batch size or identifier used in folder naming; defaults to None.
    - desc (str): Description or experiment name used in folder naming; defaults to an empty string.

    The function creates a directory if it does not exist and saves the state dictionaries 
    of the model, optimizer, and scheduler into a single file under the constructed path.
    """
    folder = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, file_path)



def load_model(modelname, dims, n_windows, batch=None, item_name="", desc=""):
    """
    Loads or initializes a model along with its optimizer and scheduler.

    Parameters:
    - modelname (str): Name of the model class to be loaded from `src.models`.
    - dims (int): Dimensionality of the input features.
    - n_windows (int): Size of the window used in data processing for models requiring it.
    - batch (optional, int or str): Batch size or identifier used in folder naming; defaults to None.
    - item_name (str): Identifier for the specific dataset item being processed.
    - desc (str): Description or experiment name used in folder naming; defaults to an empty string.

    Returns:
    - model (torch.nn.Module): The loaded or newly created neural network model.
    - optimizer (torch.optim.Optimizer): Optimizer associated with the model.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler for training.
    - is_history (bool): Indicates whether a pre-trained model was loaded (`True`) or a new one was created (`False`).
    
    Notes:
    - If a checkpoint file exists at the specified path and conditions allow loading (e.g., not retraining or in test mode),
      the function loads the saved model, optimizer, and scheduler states.
    - Otherwise, it creates a new model instance and initializes a fresh optimizer and scheduler.
    """
    model = None
    optimizer = None
    scheduler = None
    try:

        model_class = getattr(src.models, modelname)
        if modelname in ["HAO_E", "HAO_P", "HAO_H",]:
            model = model_class(dims, n_windows).double()
        else:
            model = model_class(dims).double()

        # Initialize optimizer with AdamW as default; learning rate and weight decay are hardcoded
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)

        # Initialize learning rate scheduler with StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        
        # Construct path to save/load model checkpoint
        fname = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/model.ckpt'
        
        # Check if pre-trained model exists and should be loaded
        if os.path.exists(fname) and (not args.retrain or args.test):
            print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        else:
            print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
    except Exception as e:
        print(e)
    return model, optimizer, scheduler


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True, adj_save=""):
    """
    Performs backpropagation for training or evaluation.

    Parameters:
    - epoch: Current epoch number.
    - model: The model to be trained or evaluated.
    - data: The input data.
    - dataO: Original input data, used for comparison or loss calculation.
    - optimizer: The optimizer for updating model parameters.
    - scheduler: The learning rate scheduler.
    - training: Boolean flag indicating whether it is in training mode. Default is True.
    - adj_save: Path to save adjacency matrix and curvature data.

    Returns:
    - If in training mode, returns the average loss and learning rate.
    - If not in training mode, returns the loss, predictions, and adjacency matrix and curvature differences.
    """
    feats = dataO
    if model.name in ["HAO_E", "HAO_P", "HAO_H"]:
        # Define loss functions
        l = nn.MSELoss(reduction='none')
        l_s = nn.MSELoss(reduction='none')
        l1s = []
        if training:
            # Initialize a dictionary to save curvature and loss data during training
            save_curv_dict = {
                "epoch": [],
                "init_curv": [],
                "ae_curv": [],
                "t_hgcn_in_curv": [],
                "t_hgcn_out_curv": [],
                "s_hgcn_in_curv": [],
                "s_hgcn_out_curv": [],
                "out_curv": [],
                "rec_loss": [],
                "T_loss": [],
                "S_loss": [],
                "loss": [],
            }
            try:
                # Load historical training data if exists
                with open(adj_save[2], 'r', encoding='utf-8') as f:
                    save_curv_dict = json.load(f)
                    print("Training history data loading completed")
            except Exception as e:
                print(e)

            epoch = []
            losss = torch.tensor([0]).to(device).to(torch.float64)
            t_adj_q = None
            s_adj_q = None
            t_adj = None
            s_adj = None
            t_adj_list = []
            s_adj_list = []
            history_t = None
            history_s = None
            try:
                # Load historical adjacency matrices
                t_adj_q = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
                s_adj_q = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
            except Exception as e:
                print("Historical natural structure loading failed")
                print(traceback.print_exc())
            for i, d in enumerate(data):
                # Forward pass
                x, t_adj_q, s_adj_q, curv, t_adj, s_adj = model(d, t_adj_q, s_adj_q)
                if len(t_adj_q.shape) > 2:
                    t_adj_q = t_adj_q.squeeze(dim=2)
                    s_adj_q = s_adj_q.squeeze(dim=2)
                t_adj_list.append(t_adj_q)
                s_adj_list.append(s_adj_q)
                curv_list = curv.cpu().detach().numpy()
                # Save curvature data
                save_curv_dict["init_curv"].append(float(curv_list[0][0]))
                save_curv_dict["epoch"].append(float(len(save_curv_dict["epoch"])) + 1)
                save_curv_dict["ae_curv"].append(float(curv_list[1][0]))
                save_curv_dict["t_hgcn_in_curv"].append(float(curv_list[2][0]))
                save_curv_dict["t_hgcn_out_curv"].append(float(curv_list[3][0]))
                save_curv_dict["s_hgcn_in_curv"].append(float(curv_list[4][0]))
                save_curv_dict["s_hgcn_out_curv"].append(float(curv_list[5][0]))
                save_curv_dict["out_curv"].append(float(curv_list[6][0]))
                if len(t_adj_list) < 2:
                    history_t = t_adj_list[0]
                    history_s = s_adj_list[0]
                else:
                    history_t = t_adj_list[0] - (t_adj_list[0] - t_adj_list[1]) / i
                    history_s = s_adj_list[0] - (s_adj_list[0] - s_adj_list[1]) / i
                t_adj_list = [nn.Parameter(history_t, requires_grad=False).to(device).to(torch.float64)]
                s_adj_list = [nn.Parameter(history_s, requires_grad=False).to(device).to(torch.float64)]

                # Calculate losses
                T_loss = torch.mean(l_s(history_t, t_adj_q))
                S_loss = torch.mean(l_s(history_s, s_adj_q))
                rec_loss = torch.mean(l(x, d))
                losss = rec_loss + T_loss + S_loss
                # Save loss data
                save_curv_dict["rec_loss"].append(float(rec_loss.cpu().detach().numpy()))
                save_curv_dict["T_loss"].append(float(T_loss.cpu().detach().numpy()))
                save_curv_dict["S_loss"].append(float(S_loss.cpu().detach().numpy()))
                save_curv_dict["loss"].append(float(losss.cpu().detach().numpy()))
                l1s.append(losss.item())
                optimizer.zero_grad()
                # Backward pass and optimize
                losss.backward()
                optimizer.step()
                t_adj_q = history_t
                s_adj_q = history_s
            scheduler.step()
            # Save updated adjacency matrices and training history
            np.save(adj_save[0], history_t.cpu().detach().numpy())
            np.save(adj_save[1], history_s.cpu().detach().numpy())
            with open(adj_save[2], 'w', encoding='utf-8') as f:
                json.dump(save_curv_dict, f, ensure_ascii=False, indent=4)
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            try:
                del save_curv_dict
                gc.collect()
                print("save_curv_dict is released")
            except Exception as e:
                print(e)
                print(traceback.print_exc())
            return np.mean(l1s), optimizer.param_groups[0]['lr']

        else:
            # Evaluation mode
            if model.name in ["HAO_E", "HAO_P", "HAO_H"]:
                xs = []
                adj_list_t = []
                adj_list_s = []
                fro_diff = []
                t_adj = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
                s_adj = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
                for i, d in enumerate(data):
                    x, t, s, _, _, _ = model(d.to(device), t_adj, s_adj)
                    # Calculate Frobenius norm difference
                    frobenius_norm_A = np.linalg.norm(t_adj.cpu().detach().numpy(), 'fro')
                    frobenius_norm_B = np.linalg.norm(t.cpu().detach().numpy(), 'fro')
                    coff_t = np.abs(frobenius_norm_A - frobenius_norm_B)
                    frobenius_norm_A = np.linalg.norm(s_adj.cpu().detach().numpy(), 'fro')
                    frobenius_norm_B = np.linalg.norm(s.cpu().detach().numpy(), 'fro')
                    coff_s = np.abs(frobenius_norm_A - frobenius_norm_B)
                    fro_diff.append([2 * (coff_s * coff_t) / (coff_s + coff_t + 0.00001), coff_s, coff_t])
                    xs.append(x)
                fro_mid = np.sum(np.expand_dims(np.asarray(fro_diff)[:, 0], axis=1), axis=1)
                pos_list = [np.argmax(fro_mid), np.argmin(fro_mid)]
                if model.name in ["HAO_E", "HAO_P", "HAO_H"]:
                    for pos in pos_list:
                        x, t, s, _, _, _ = model(data[pos].to(device), t_adj, s_adj)
                        adj_list_t.append(t.cpu().detach().numpy())
                        adj_list_s.append(s.cpu().detach().numpy())
                xs = torch.stack(xs)
                y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                loss = l(xs.to(device), data.to(device))
                loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                try:
                    del xs
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("xs released")
                except:
                    print("xs release failed")
                    print(traceback.print_exc())
                return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), (adj_list_t, adj_list_s, fro_diff)
            

def trainModel(model_name, dataset_name, epoch, windows, batch_size, item_dataSet, data_rate=0.9,desc=""):
    """
    Trains a model on the specified dataset with given parameters.

    Parameters:
    - model_name (str): Name of the model to be trained.
    - dataset_name (str): Name of the dataset used for training.
    - epoch (int): Current epoch number, used in training loop.
    - windows (int): Window size used in data processing.
    - batch_size (int): Size of the batch used in training.
    - item_dataSet (str): Identifier for the specific dataset item being processed.
    - desc (str): Description or experiment name used in folder naming; defaults to an empty string.

    The function handles the complete training process including model initialization, 
    data loading, training loop, and evaluation. It also supports saving model checkpoints 
    and performing post-training analysis based on structural differences in graph models.
    """
    # Initialize batch size
    batch_size = batch_size
    # Assign model name to local variable
    model = model_name
    # Assign dataset name to local variable
    dataset = dataset_name
    # Set model and dataset in global args for reference elsewhere
    args.model = model
    args.dataset = dataset
    # Store new epoch count for training continuation
    new_epoch = epoch
    # Initialize model, optimizer, and scheduler
    model, optimizer, scheduler = None, None, None
    
    # Define directory path for saving training records and intermediate results
    folder = f'trainRecord/{model_name}_{args.dataset}/{item_dataSet}/{batch_size}/{desc}-{windows}/'
    os.makedirs(folder, exist_ok=True)
    # Paths for saving adjacency matrices and curvature data during training
    train_state_adj = [
        f'{folder}/{model_name}_t_adj.npy', 
        f'{folder}/{model_name}_s_adj.npy',
        f'{folder}/{model_name}_curv_{batch_size}.json'
    ]
    
    # Data sampling rate for subsequent analysis
    Data_Rate = data_rate

    # Load or initialize the model along with optimizer and scheduler
    model, optimizer, scheduler = load_model(
        args.model, features, n_windows=windows,
        desc=desc, batch=batch_size, item_name=item_dataSet
    )
    
    # Initialize data loader for training stage
    data_loader = dataLoader(
        args.dataset, modelName=model_name, convertWindow=windows, stage="Train",
        item_dataSet=item_dataSet
    )
    
    # Get feature dimension from the dataset
    features = data_loader.getDim()
    # Move model to the designated device (e.g., GPU)
    model.to(device)

    # DataLoader for training data with shuffling enabled
    trainStage = DataLoader(data_loader, batch_size=len(data_loader), shuffle=True)
    # DataLoader for test data without shuffling
    test_loader = dataLoader(
        args.dataset, modelName=model_name, convertWindow=windows, stage="Test", 
        item_dataSet=item_dataSet
    )
    testStage = DataLoader(test_loader, batch_size=len(test_loader), shuffle=False)

    # Proceed only if not in test mode
    if not args.test:        
        # Training loop over epochs
        for e in tqdm(list(range(epoch + 1, epoch + new_epoch + 1))):
            model.train()  
            train_data = None
            print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
            
            # Iterate through training data
            for train_data in trainStage:
                # Perform backpropagation step
                lossT, lr = backprop(e, model, train_data, features, optimizer, scheduler, training=True,
                                     adj_save=train_state_adj)
                
                # Save model checkpoint
                save_model(model, optimizer, scheduler, windows, item_name=item_dataSet,
                           desc=desc, batch=batch_size)
                break
            
            # Clean up memory after training phase
            try:
                del train_data
                gc.collect()
                torch.cuda.empty_cache()
                print("Release test phase data")
            except Exception as e:
                print("Failed to release test phase data")
                print(traceback.print_exc())

            # Enable zero gradient flag
            torch.zero_grad = True
            # Switch to evaluation mode
            model.eval()
            with torch.no_grad():  
                print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

                # Iterate through test data
                for test, label in testStage:
                    try:
                        # Process labels to binary format
                        labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                        labels_counts = torch.bincount(torch.tensor(labelsFinal))
                    except:
                        labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                        labelsFinal = (np.sum(labelsFinal, axis=1) >= 1) + 0
                        labels_counts = torch.bincount(torch.tensor(labelsFinal))

                    # Skip if no positive or negative samples are found
                    if len(labels_counts) < 2:
                        print(f'{color.HEADER} all is pos samplers {args.dataset}{color.ENDC}')
                        continue
                    if labels_counts[1] < 2:
                        print(f'{color.HEADER} all is neg samplers < 2{args.dataset}{color.ENDC}')
                        continue

                    # Perform backpropagation in evaluation mode
                    loss, y_preds, adj = backprop(0, model, test, features, optimizer, scheduler, training=False,
                                                  adj_save=train_state_adj)
                    clean_test = test.cpu().detach().numpy()
                    test_shape = test.shape[0]

                    # Clean up memory after test data usage
                    try:
                        del test
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Release test data")
                    except Exception as e:
                        print("Failed to release test data")
                        print(traceback.print_exc())

                    # Compute final loss
                    lossFinal = np.mean(loss, axis=1)

                    # Structural difference measurement for specific models
                    if model_name in ["HAO_E", "HAO_P", "HAO_H"]:
                        # Sampling logic for positive and negative indices
                        samper_num_pos = int(len(pos_test_index[0]) * Data_Rate)
                        samper_num_neg = int(len(neg_test_index[0]) * Data_Rate)
                        all_index = set(list(pos_test_index[0])).union(set(list(neg_test_index[0])))

                        # Prepare labels and predictions for evaluation
                        labelY = label.cpu().detach().numpy()
                        if labelY.shape[1] != features:
                            labelY = labelY.reshape(test_shape, -1, features)[:, -1, :]
                        labelY = np.concatenate(
                            [labelY, np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape)],
                            axis=1
                        )
                        train_pre = clean_test.reshape(test_shape, -1, features)[:, -1, :]
                        neg_test_index = np.where(labelsFinal > 0)
                        pos_test_index = np.where(labelsFinal == 0)
                        Train_pre = np.concatenate([train_pre, np.expand_dims(np.asarray(adj[2])[:, 0], axis=1)],
                                                    axis=1)
                        Y_preds = np.concatenate(
                            [y_preds, np.zeros(shape=np.expand_dims(np.asarray(adj[2])[:, 0], axis=1).shape)],
                            axis=1
                        )

                        # Sample negative and positive indices
                        sampler_neg = np.random.choice(neg_test_index[0], size=samper_num_neg, replace=False)
                        sampler_pos = np.random.choice(pos_test_index[0], size=samper_num_pos, replace=False)

                        # Prepare training and testing static data for evaluation
                        train_static = np.concatenate([Train_pre[sampler_neg], Train_pre[sampler_pos]], axis=0)
                        train_static_label = np.concatenate([labelY[sampler_neg], labelY[sampler_pos]], axis=0)
                        test_static = Train_pre[list(all_index - set(sampler_neg) - set(sampler_pos))]
                        test_static_label = labelY[list(all_index - set(sampler_neg) - set(sampler_pos))]
                        predict_t, predict_s = np.concatenate([Y_preds[sampler_neg], Y_preds[sampler_pos]],
                                                              axis=0), Y_preds[
                            list(all_index - set(sampler_neg) - set(sampler_pos))]

                        # Structure data for validation metrics calculation
                        s_train_struct = [
                            train_static_label,
                            predict_t,
                            train_static,
                        ]
                        s_val_struct = [
                            test_static_label,
                            predict_s,
                            test_static
                        ]

                        # Calculate optimal metrics
                        _, optimal_metrics, _ = get_val_res(scores=None, labels=None, th=None, path=None,
                                                           search_res=None, test_data=s_train_struct,
                                                           val_data=s_val_struct)

                        # Memory cleanup after evaluation
                        try:
                            del s_train_struct, s_val_struct, Y_preds
                            gc.collect()
                        except Exception as e:
                            print(e)
                            print(traceback.print_exc())
                        
                        # Print evaluation metrics
                        print("Graph structure difference measurement:", optimal_metrics["f1"])
                        print("Graph structure difference measurement:", optimal_metrics["roc_auc"])
                        print("Graph structure difference measurement:", optimal_metrics["precision"])
                        print("Graph structure difference measurement:", optimal_metrics["recall"])
                        print("Graph structure difference measurement:", optimal_metrics["accuracy"])

                # Clean up memory after label and loss handling
                try:
                    del label, lossFinal, adj
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(e)
                    traceback.print_stack()

    # Final memory cleanup
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        traceback.print_stack()



def getDataSetList(dataSet):
    """
    Generates a list of dataset identifiers based on the specified dataset name.

    This function is used to determine which specific subsets or items of a dataset 
    should be processed during training or evaluation. The returned list contains 
    identifiers for each item in the dataset, which are typically used to load 
    corresponding data files.

    Parameters:
    - dataSet (str): Name of the dataset. Valid options include 'ASD', 'MSL', 'SMAP', and 'SMD'. 
                     For other datasets, it dynamically generates identifiers from available files.

    Returns:
    - list: A list of string identifiers for the dataset items. These identifiers correspond 
            to prefixes of file names in the processed data directory for the given dataset.

    Raises:
    - Exception: If the processed data folder for the dataset does not exist.
    """
    req = []
    import glob
    import re

    # Construct path to the dataset's processed data folder
    folder = os.path.join(output_folder, dataSet)

    # Check if the folder exists
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')

    # Find all training files in the folder
    files = glob.glob(os.path.join(folder, "*train.npy"))

    # Define fixed dataset item identifiers based on dataset name
    if dataSet == "ASD":
        req = ["omi-10_"]
    elif dataSet == "MSL":
        req = ["C-1_"]
    elif dataSet == "SMAP":
        req = ["A-4_"]
    elif dataSet == "SMD":
        req = ["machine-3-7_"]
    else:
        # Dynamically generate identifiers for unknown datasets
        for file in files:
            # Extract identifier by removing folder path and file extension
            identifier = file[len(folder) + 1:len(file) - len("_train.npy") + 1]
            req.append(identifier)

    return req


def getTrainHistory(model, epoch_mod, dataset, batch=None, window="", desc=""):
    """
    Retrieves training history for a specific model and dataset to determine how many more epochs are needed.

    This function checks the existing training records to find out how many epochs have already been completed 
    for each item in the dataset. Based on this information, it calculates the remaining number of epochs required 
    to reach the target number of training iterations (`epoch_mod`).

    Parameters:
    - model (str): Name of the model whose training history is being checked.
    - epoch_mod (int): Target number of training epochs; used to calculate remaining epochs.
    - dataset (str): Name of the dataset used for training.
    - batch (optional, int or str): Batch size or identifier used in folder naming; defaults to None.
    - window (optional, str): Window size used in data processing; defaults to an empty string.
    - desc (optional, str): Description or experiment name used in folder naming; defaults to an empty string.

    Returns:
    - tuple: 
        - list: A list indicating how many additional epochs are needed for each dataset item.
        - list: A list of dataset items corresponding to the specified dataset.

    Raises:
    - Exception: If the training record directory does not exist.
    """

    # Define path to training records
    history_path = os.getcwd() + "/trainRecord"
    
    # Check if training record directory exists
    if not os.path.exists(history_path):
        raise Exception('Processed Data not found.')

    # Get list of dataset items using [getDataSetList]
    need_train = getDataSetList(dataSet=dataset)
    
    # Initialize list to track required epochs for each item
    need_train_epoch = [epoch_mod for item in need_train]

    # Iterate over dataset items to check training history
    for index, n in enumerate(need_train):
        file_list = glob.glob(os.path.join(history_path, "{}_{}/{}/".format(model, dataset, n)))
        
        # Process each training record
        for item in file_list:
            epoch = []
            # Construct path to training record file
            item_path = item + "/{}/{}-{}/model-{}_{}.mat".format(batch, desc, window, model, batch)
            
            try:
                # Load training history from .mat file
                data = sio.loadmat(item_path)
                # Extract recorded epochs
                epoch = data["epochr"].tolist()[0]
            except:
                # Default to empty list if no epoch data is found
                epoch = []

            # Calculate remaining epochs needed
            p = epoch_mod - len(epoch)
            if p > 0:
                need_train_epoch[index] = p  # Update with remaining epochs
            else:
                need_train_epoch[index] = 0  # No further training needed

    return need_train_epoch, need_train



if __name__ == '__main__':
    """
    Main execution block when this script is run directly.

    This section parses command-line arguments and initiates a training workflow for deep learning models.
    It configures the dataset, model, window size, and other training parameters. Then it determines how many epochs 
    are still needed based on previous training history and starts the training process accordingly.

    The loop iterates through combinations of batch sizes, models, datasets, and window sizes.
    For each combination, it retrieves training history using [getTrainHistory]
    and runs training via [trainModel].

    Expected command-line arguments:
    - `args.dataset`: Name of the dataset (e.g., 'SMD', 'SMAP').
    - `args.model`: Name of the model class to be used.
    - `args.windowsize`: Size of the sliding window for sequence processing.
    - `args.epoch`: Total number of epochs to train.
    - `args.space`: Extra identifier for saving results.
    - `args.data_rate`: Sampling rate used during structural difference measurement in evaluation.

    Note: Assumes that `args` has been parsed earlier in the code.
    """

    # Capture command-line arguments passed to the script (excluding the script name itself)
    commands = sys.argv[1:]

    # Dataset to be used for training; taken from command-line arguments
    Dataset = [args.dataset]

    # Model to be trained; taken from command-line arguments
    models = [args.model]

    # Window size used in data processing; taken from command-line arguments
    WindowSize = [args.windowsize]

    # Total number of training epochs
    epoch = args.epoch

    # Batch size configuration (currently only one value is used: [1])
    batch_size = [1]

    # Data sampling rate used in graph structure analysis during evaluation
    data_rate = args.dataRate

    # Description or identifier for the experiment, composed of model name and extra space parameter
    desc = args.model + "-" + args.space

    # Iterate over all combinations of configurations and start training
    for b in batch_size:
        for m in models:
            for d in Dataset:
                for window in WindowSize:
                    # Determine required number of additional epochs per dataset item
                    need_train_epoch, data_list = getTrainHistory(
                        model=m,
                        epoch_mod=epoch,
                        dataset=d,
                        batch=b,
                        window=window,
                        desc=desc
                    )

                    # Run training for the determined number of epochs
                    for e in enumerate(range(epoch)):
                        # Start the training process
                        trainModel(
                            model_name=m,
                            dataset_name=d,
                            epoch=1,
                            windows=window,
                            batch_size=b,
                            item_dataSet=getDataSetList(dataSet=d)[0],  # Use the first dataset item
                            data_rate=data_rate,
                            desc=desc
                        )
    
