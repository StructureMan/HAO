import glob
import os
import random
import networkx as nx
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.backends.backend_pgf import PdfPages
from src.folderconstants import output_folder
import matplotlib.pyplot as plt


def plotPried(true, predict, threshould, asd, labels, p_true_labels, p_labels="", dataname="", wholeData="", f1=0.0,
              stand=0.0, Train_name="", item_name="", pot_f1=0.0, s_f1=0.0, rec_loss=[], t_adj="", s_adj="", t_adj_t=[],
              s_adj_t=[], rec_loss_h=[], Struct_mess=None, best_up=None, splite_data=0.1):
    """
    Plot and save visualizations of validation data, reconstruction data, and anomaly detection results.

    Args:
        true: Ground truth validation data (time series).
        predict: Reconstructed data from the model.
        threshould: Threshold value for anomaly detection.
        asd: Additional structured data (not used in this function).
        labels: Ground truth labels for anomalies.
        p_true_labels: Predicted true labels (not used in this function).
        p_labels: Placeholder for predicted labels (not used in this function).
        dataname: Name of the dataset (used for saving files).
        wholeData: Placeholder for full dataset (not used in this function).
        f1: F1 score (not used in this function).
        stand: Standard deviation or related metric (not used in this function).
        Train_name: Training dataset name (not used in this function).
        item_name: Item name (not used in this function).
        pot_f1: POT-based F1 score (not used in this function).
        s_f1: Structured F1 score (not used in this function).
        rec_loss: Reconstruction loss (not used in this function).
        t_adj: Placeholder for temporal adjustment (used for conditional plotting).
        s_adj: Placeholder for spatial adjustment (not used in this function).
        t_adj_t: Temporal adjustment data (not used in this function).
        s_adj_t: Spatial adjustment data (not used in this function).
        rec_loss_h: Reconstruction loss history (not used in this function).
        Struct_mess: Structured message containing validation data and predictions.
        best_up: Best update information (not used in this function).
        splite_data: Data split ratio (not used in this function).

    Returns:
        None: Saves plots and numpy files to disk.
    """
    plt.clf()  # Clear the current figure
    num_subfig = true.shape[1] + 5  # Total number of subplots
    plt.figure(figsize=(30, 3 * num_subfig))  # Set figure size

    # Determine if structured validation data is 1D or 2D
    is_div = len(np.asarray(Struct_mess["Struct_val_gt"]).shape)

    # Initialize arrays for validation labels and predictions
    labels_val = np.zeros(shape=(true.shape[0]))
    labels_val_pre = np.zeros(shape=(true.shape[0]))
    val_index = Struct_mess["gt_label_val_index"]
    val_gt = Struct_mess["Struct_val_gt"]
    val_pred = Struct_mess["Struct_val_pred"]

    # Map validation ground truth and predictions to their respective indices
    for index, item in enumerate(val_index):
        labels_val[item] = val_gt[index]
        labels_val_pre[item] = val_pred[index]

    # Plot validation data, reconstruction data, and ground truth for each sensor
    for item_fig in range(1, true.shape[1] + 1):
        plt.subplot(num_subfig, 1, item_fig)
        plt.plot(true.T[item_fig - 1], c="red", alpha=0.6, label="Validation data", linewidth=1.5)
        plt.plot(predict.T[item_fig - 1], c="green", alpha=0.6, label="Reconstruction data", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]), labels.T[item_fig - 1], color='#F48222', alpha=0.9,
                         label=f"Ground truth\n sensor:{item_fig}")

        # Add validation truth and predictions if temporal adjustment is enabled
        if t_adj is not None:
            if is_div == 2:
                plt.fill_between(np.arange(labels.shape[0]),
                                 np.asarray(Struct_mess["Struct_val_gt"]).T[item_fig - 1],
                                 color='yellow', alpha=0.2, label="Val truth")
                plt.fill_between(np.arange(labels.shape[0]),
                                 np.asarray(Struct_mess["fine_pred_label"]).T[item_fig - 1],
                                 color='blue', alpha=0.1, label="Val Pre")
            else:
                plt.fill_between(np.arange(labels.shape[0]),
                                 np.asarray(Struct_mess["fine_pred_label"]).T[item_fig - 1],
                                 color='blue', alpha=0.9, label="Pred Label")

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot

    # Plot additional structured metrics if temporal adjustment is enabled
    if t_adj is not None:
        # Plot reconstruction difference
        plt.subplot(num_subfig, 1, true.shape[1] + 1)
        plt.plot(Struct_mess["Struct_rdiff"], c="green", alpha=0.6, label="Rec Diff", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]), labels.T[0] * Struct_mess["Struct_rdiff"],
                         color='red', alpha=0.3, label="Ground truth")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot structural difference
        plt.subplot(num_subfig, 1, true.shape[1] + 2)
        plt.plot(Struct_mess["Struct_sdiff"], c="green", alpha=0.6, label="Struct_sdiff", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]), labels.T[0] * Struct_mess["Struct_sdiff"],
                         color='red', alpha=0.3, label="Ground truth")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot temporal difference
        plt.subplot(num_subfig, 1, true.shape[1] + 3)
        plt.plot(Struct_mess["Struct_tdiff"], c="green", alpha=0.6, label="Struct_tdiff", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]), labels.T[0] * Struct_mess["Struct_tdiff"],
                         color='red', alpha=0.3, label="Ground truth")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot anomaly score
        plt.subplot(num_subfig, 1, true.shape[1] + 4)
        plt.plot(Struct_mess["Struct_anomaly"], c="green", alpha=0.6, label="Anomaly Score", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]),
                         np.asarray(Struct_mess["Struct_ture"]) * max(Struct_mess["Struct_anomaly"]),
                         color='red', alpha=0.3, label="Ground truth")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot predicted labels
        plt.subplot(num_subfig, 1, true.shape[1] + 5)
        fine_pred_label = np.asarray(Struct_mess["fine_pred_label"])
        if len(fine_pred_label.shape) > 1:
            fine_pred_label = np.sum(fine_pred_label, axis=1) > 0
            fine_pred_label = (fine_pred_label > 0).astype(int)
        plt.plot(Struct_mess["Struct_anomaly"], c="green", alpha=0.6, label="Anomaly Score", linewidth=1.5)
        plt.fill_between(np.arange(labels.shape[0]), fine_pred_label, color='blue',
                         alpha=1.0, label="Pred Label")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Save structured predictions and labels as numpy files
        np.save(f"{dataname}_Correctlabel.npy", np.asarray(Struct_mess["y_pred"]), allow_pickle=True)
        np.save(f"{dataname}_Correctfinelabel.npy", np.asarray(Struct_mess["fine_pred_label"]), allow_pickle=True)

    # Save the final plot as a PNG file and close the figure
    plt.savefig('{}.png'.format(dataname))
    plt.close()

