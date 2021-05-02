
"""
This file contains code to compute metrics (accuracy or others as required) and ploting confusion matrix and loss curves.
"""

import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    cf_matrix = confusion_matrix(labels, preds)
    # results["macro_precision"], results["macro_recall"], results[
    #     "macro_f1"], _ = precision_recall_fscore_support(
    #     labels, preds, average="macro")
    # results["micro_precision"], results["micro_recall"], results[
    #     "micro_f1"], _ = precision_recall_fscore_support(
    #     labels, preds, average="micro")
    # results["weighted_precision"], results["weighted_recall"], results[
    #     "weighted_f1"], _ = precision_recall_fscore_support(
    #     labels, preds, average="weighted")

    return results, cf_matrix

def plot_confusion_matrix(cf_matrix, norm_cf_matrix, labels, output_dir, mode, epoch):
    annot_kws={'size': 6}
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.tick_params(axis='both', labelsize=7)
    sns.heatmap(cf_matrix, annot=True, annot_kws=annot_kws, fmt='', xticklabels=labels, yticklabels=labels, square=False, ax=ax)
    plt.savefig(os.path.join(output_dir, "{}-cf_matrix_count-{}.svg".format(mode, epoch)), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.tick_params(axis='both', labelsize=7)
    sns.heatmap(norm_cf_matrix, annot=True, annot_kws=annot_kws, fmt='.2', cmap='Blues', xticklabels=labels, yticklabels=labels, square=False, ax=ax)
    plt.savefig(os.path.join(output_dir, "{}-cf_matrix_acc-{}.svg".format(mode, epoch)), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def save_plot(l1, l2, output_dir, save_name, label="label", xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.tick_params(axis='both', labelsize=6)       
    font = {'size': 6}
    matplotlib.rc('font', **font)
    plt.plot(l1, l2, label=label)
    for i, t in enumerate(l2):
        plt.text(i, t, t, horizontalalignment='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{save_name}.png"), format='png')
    plt.close()
