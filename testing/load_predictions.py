
import os
import numpy as np
import pandas as pd
import math

import sys
sys.path.append("/users/ak1774/scratch/esport/death_prediction")


import preprocess
import data_loader
import models
import test_model

import commentjson
from pydoc import locate

import pickle
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def load_all_presictions():

    all_result_files = glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_pred_results/test_results/*.pickle")
    all_model_path_file = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_pred_results/test_results/model_path_list.pickle"

    all_y_files = [ filename for filename in all_result_files if "all_y" in filename]


    all_result_filepath_tuples = []

    # find matching predictions
    for all_y_path in all_y_files:
        folder,filename = os.path.split(all_y_path)
        
        all_pred_filename = filename.replace("all_y","all_pred")
        all_pred_path = os.path.join(folder,all_pred_filename)


        per_sec_filename = filename.replace("all_y","per_sec_pred")
        per_sec_path = os.path.join(folder,per_sec_filename)

        all_result_filepath_tuples.append((all_y_path,all_pred_path,per_sec_path))


    #print(all_result_filepath_tuples)

    with open(all_model_path_file, 'rb') as f:
        all_model_path = pickle.load(f)


    all_y = [ [] for model_path in all_model_path]
    all_pred = [ [] for model_path in all_model_path]


    all_per_sec_pred = [ [[] for _ in range(20)] for model_path in all_model_path ]

    all_result_filepath_tuples = all_result_filepath_tuples[:]
    for ooo,(y_path,pred_path,per_sec_path) in enumerate(all_result_filepath_tuples):
        print(ooo)
        with open(y_path, 'rb') as f:
            y = pickle.load(f)

        with open(pred_path, 'rb') as f:
            pred = pickle.load(f)

        with open(per_sec_path, 'rb') as f:
            per_sec_pred = pickle.load(f)

        for model_i in range(len(all_model_path)):
            all_y[model_i].extend(y[model_i])
            all_pred[model_i].extend(pred[model_i])
            for timeslot_i in range(20):
                all_per_sec_pred[model_i][timeslot_i].extend(per_sec_pred[model_i][timeslot_i])

    #print("all_y.shape ",all_y[0].shape)
    #print("all_pred.shape ",all_pred[0].shape)
    #print("all_per_sec_pred.shape ",len(all_per_sec_pred[0][0]))

    return all_y,all_pred,all_per_sec_pred,all_model_path






