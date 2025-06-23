import glob
import json
import datetime
import os
import sys

import torch
from tqdm import tqdm
import numpy as np
import optimizers
from src.evaluate import *
from src.folderconstants import output_folder
from src.models import *
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn as nn
from time import time
from scipy.io import savemat
import scipy.io as sio

from src.parser import args
from src.utils import color
from utils.data_utils import cacluateJaccard
import time

device = "cuda:1"
import warnings
import pandas as pd

debug = False
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


class dataLoader(Dataset):
    def __init__(self, DataName="", modelName="", convertWindow="", stage="Train", item_dataSet=""):
        self.DataName = DataName
        self.modelName = modelName
        self.convertWindow = convertWindow
        self.stage = stage
        self.limitModel = ["HAO"]
        self.limitStackModel = []
        exception_models = ["HAO"]
        folder = os.path.join(output_folder, self.DataName)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        self.loader = []
        for file in ['train', 'test', 'labels']:
            if DataName == 'ASD': file = item_dataSet + file
            if DataName == 'SMD': file = item_dataSet + file
            if DataName == 'SMAP': file = item_dataSet + file
            if DataName == 'MSL': file = item_dataSet + file
            self.loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True)[20:])
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
        return self.loader[0].shape[1]

    def convertWindowPro(self, index):
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
                        w.contiguous().view(-1))
            return torch.stack(windows)

    def __len__(self):
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

    def construct_t_adj(self, data):
        data = np.reshape(data, (self.convertWindow, -1))
        shape = data.shape
        distances = torch.pdist(data, p=2).to(device)
        adjacency_matrix = torch.zeros((shape[0], shape[0]), dtype=torch.float64).to(device)
        index = torch.triu_indices(shape[0], shape[0], offset=1).to(device)
        adjacency_matrix[index[0], index[1]] = distances.to(torch.float64)
        # adjacency_matrix = adjacency_matrix + adjacency_matrix.t()  # 保证邻接矩阵是对称的
        return adjacency_matrix.to(device)

    def construct_s_adj(self, data):
        data = np.reshape(data, (-1, self.convertWindow))
        shape = data.shape
        distances = torch.pdist(data, p=2).to(device)
        adjacency_matrix = torch.zeros((shape[0], shape[0]), dtype=torch.float64).to(device)
        index = torch.triu_indices(shape[0], shape[0], offset=1).to(device)
        adjacency_matrix[index[0], index[1]] = distances.to(torch.float64)
        # adjacency_matrix = adjacency_matrix + adjacency_matrix.t()  # 保证邻接矩阵是对称的
        return adjacency_matrix.to(device)

    def __getitem__(self, item):
        if self.stage == "Train":
            if self.modelName in self.limitModel:
                # return [torch.FloatTensor(self.trainData[item]).to(device).to(torch.float64), self.construct_t_adj(
                #     self.trainData[item]), self.construct_s_adj(self.trainData[item].t())]
                return torch.FloatTensor(self.trainData[item]).to(device).to(torch.float64)
            else:
                # return [torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)]
                return torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)
        else:
            if self.modelName in self.limitModel:
                # return [torch.FloatTensor(self.testData[item]).to(device).to(torch.float64), torch.FloatTensor(
                #     self.labelData[item]).to(
                #     device).to(torch.float64), self.construct_t_adj(self.trainData[item]), self.construct_s_adj(
                #     self.trainData[item].t())]
                return torch.FloatTensor(self.testData[item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.labelData[item]).to(
                    device).to(torch.float64)
            else:
                # return [torch.FloatTensor(self.loader[1][item]).to(device).to(torch.float64), torch.FloatTensor(
                #     self.loader[2][item]).to(
                #     device).to(torch.float64)]
                return torch.FloatTensor(self.loader[1][item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.loader[2][item]).to(
                    device).to(torch.float64)


def save_model(model, optimizer, scheduler, epoch, accuracy_list, n_windows, item_name, batch=None, desc=""):
    folder = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims, n_windows, batch=None, item_name="", desc=""):
    import src.models
    model_class = getattr(src.models, modelname,args.space)
    if modelname in ["HAO"]:
        model = model_class(dims, n_windows, args.space).double()

    optimizer = getattr(optimizers, "RiemannianAdam")(params=model.parameters(), lr=model.lr,
                                                      weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    fname = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True, adj_save=""):
    global loss
    feats = dataO
    if model.name == "HAO":
        l = nn.MSELoss(reduction='none')
        l1s = []
        if training:
            batch_num = 256
            losss = torch.tensor([0]).to(device).to(torch.float64)
            t_adj_q = None
            s_adj_q = None
            for i, d in enumerate(data):
                if i < batch_num:
                    x, t_adj_q, s_adj_q, curv = model(d, t_adj_q, s_adj_q)
                    np.save(adj_save[0], t_adj_q.cpu().detach().numpy())
                    np.save(adj_save[1], s_adj_q.cpu().detach().numpy())
                    losss = losss + torch.mean(l(x, d))
                else:

                    losss = losss / batch_num
                    l1s.append(losss.item())
                    tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
                    optimizer.zero_grad()
                    losss.backward()
                    optimizer.step()
                    losss = torch.tensor([0]).to(device).to(torch.float64)
                    batch_num = i + batch_num

            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            xs = []
            adj_list_t = []
            adj_list_s = []
            jacced = []
            t_adj = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
            s_adj = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
            for d in data:
                x, t, s, _ = model(d.to(device), t_adj, s_adj)
                adj_list_t.append(t)
                adj_list_s.append(s)
                coff_t = cacluateJaccard(t_adj=t_adj, test_adj=t, K=model.n_window)
                coff_s = cacluateJaccard(t_adj=s_adj, test_adj=s, K=model.n_feats)
                jacced.append(2 * (coff_s * coff_t) / (coff_s + coff_t + 0.00001))
                xs.append(x)
            adj_list_t = torch.stack(adj_list_t).cpu().detach().numpy()
            adj_list_s = torch.stack(adj_list_s).cpu().detach().numpy()
            xs = torch.stack(xs)
            jacced = torch.stack(jacced)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs.to(device), data.to(device))
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), (adj_list_t, adj_list_s, jacced)


def trainModel(model_name, dataset_name, epoch, windows, batch_size, item_dataSet, desc=""):
    batch_size = batch_size
    model = model_name
    dataset = dataset_name
    args.model = model
    args.dataset = dataset
    new_epoch = epoch
    data_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Train",
                             item_dataSet=item_dataSet)
    features = data_loader.getDim()
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, features, n_windows=windows, desc=desc,
                                                                   batch=batch_size, item_name=item_dataSet)

    model.to(device)
    trainStage = DataLoader(data_loader, batch_size=len(data_loader), shuffle=True)
    test_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Test",
                             item_dataSet=item_dataSet)
    testStage = DataLoader(test_loader, batch_size=len(test_loader), shuffle=False)

    folder = f'trainRecord/{model_name}_{args.dataset}/{item_dataSet}/{batch_size}/{desc}-{windows}/'
    os.makedirs(folder, exist_ok=True)
    train_state_adj = [f'{folder}/{model_name}_t_adj.npy', f'{folder}/{model_name}_s_adj.npy']

    # train stage
    num_epochs = new_epoch
    for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
        model.train()
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        for train in trainStage:
            lossT, lr = backprop(e, model, train, features, optimizer, scheduler, training=True,
                                 adj_save=train_state_adj)
            save_model(model, optimizer, scheduler, e, accuracy_list, windows, item_name=item_dataSet,
                       desc=desc,
                       batch=batch_size)
            model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, features, desc=desc,
                                                                           n_windows=windows, batch=batch_size,
                                                                           item_name=item_dataSet)

        torch.zero_grad = True
        model.eval()
        with torch.no_grad():
            print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
            for test, label in testStage:
                try:

                    labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                    labels_counts = torch.bincount(torch.tensor(labelsFinal))
                except:
                    labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                    labelsFinal = (np.sum(labelsFinal, axis=1) >= 1) + 0
                    labels_counts = torch.bincount(torch.tensor(labelsFinal))

                if len(labels_counts) < 2:
                    continue
                if labels_counts[1] < 2:
                    continue

                loss, y_preds, adj = backprop(0, model, test, features, optimizer, scheduler, training=False,
                                              adj_save=train_state_adj)

                lossFinal = np.mean(loss, axis=1)

                smooth_err = get_err_scores(lossFinal, labelsFinal)
                rate = 0.1

                neg_test_index = np.where(labelsFinal > 0)
                pos_test_index = np.where(labelsFinal == 0)

                samper_num = int(len(neg_test_index[0]) * rate)
                if samper_num == 0:
                    samper_num = int(len(neg_test_index[0]) / 2)
                all_index = set(list(pos_test_index[0])).union(set(list(neg_test_index[0])))
                sampler_neg = np.random.choice(neg_test_index[0], size=samper_num, replace=False)
                sampler_pos = np.random.choice(pos_test_index[0], size=samper_num, replace=False)

                find_best_data = np.concatenate([smooth_err[sampler_neg], smooth_err[sampler_pos]], axis=0)
                find_best_data_label = np.concatenate([labelsFinal[sampler_neg], labelsFinal[sampler_pos]],
                                                      axis=0)
                optimal_threshold, _ = search_optimal_threshold(find_best_data, find_best_data_label)

                overplus_data = smooth_err[list(all_index - set(sampler_neg) - set(sampler_pos))]
                overplus_data_label = labelsFinal[list(all_index - set(sampler_neg) - set(sampler_pos))]
                optimal_threshold, optimal_metrics = get_val_res(overplus_data, overplus_data_label,
                                                                 optimal_threshold)
                print(optimal_metrics)



# Obtain training data
def getDataSetList(dataSet):
    req = []
    folder = os.path.join(output_folder, dataSet)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    files = glob.glob(os.path.join(folder, "*train.npy"))
    if dataSet == "ASD":
        req = ["omi-10_", ]
    elif dataSet == "MSL":
        req = ["C-1_"]
    elif dataSet == "SMAP":
        req = ["A-4_"]
    elif dataSet == "SMD":
        req = ["machine-3-7_"]
    else:
        for file in files:
            req.append(file[len(folder) + 1:len(file) - len("_train.npy") + 1])
    return req


# Retrieve training history.
def getTrainHistory(model, epoch_mod, dataset, batch=None, window="", desc=""):
    history_path = os.getcwd() + "/trainRecord"
    if not os.path.exists(history_path):
        raise Exception('Processed Data not found.')
    need_train = getDataSetList(dataSet=dataset)
    need_train_epoch = [epoch_mod for item in need_train]
    for index, n in enumerate(need_train):
        file_list = glob.glob(os.path.join(history_path, "{}_{}/{}/".format(model, dataset, n)))
        for item in file_list:
            item_path = item + "/{}/{}-{}/model-{}_{}.mat".format(batch, desc, window, model, batch)
            try:
                data = sio.loadmat(item_path)
                epoch = data["epochr"].tolist()[0]
            except:
                epoch = []
            p = epoch_mod - len(epoch)
            if p > 0:
                need_train_epoch[index] = p
            else:
                need_train_epoch[index] = 0
    return need_train_epoch, need_train


if __name__ == '__main__':
    commands = sys.argv[1:]
    Dataset = [args.dataset]
    models = [args.model]
    WindowSize = [args.windowsize]
    epoch = args.epoch
    batch_size = [256]
    desc = args.model + "-" + args.space
    for b in batch_size:
        for m in models:
            for d in Dataset:
                for window in WindowSize:
                    need_train_epoch, data_list = getTrainHistory(model=m, epoch_mod=epoch, dataset=d, batch=b, window=window, desc=desc)
                    for e in enumerate(range(epoch)):
                            trainModel(model_name=m, dataset_name=d, epoch=1, windows=window,batch_size=b, item_dataSet=getDataSetList(dataSet=d)[0], desc=desc)

    print("test demo")