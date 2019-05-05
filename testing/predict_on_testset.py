
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



dataPathList = glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/test_whole_matches/*.h5")
#dataPathList = dataPathList[0:50]

modelPathList = [#"/mnt/lustre/groups/cs-dclabs-2019/esport/death_pred_results/2019-03-12_18-30-12/10/model.model"
                  "/users/ak1774/scratch/esport/cluster_results/FixedMinimal/3/model.model",
                  "/mnt/lustre/users/ak1774/esport/cluster_results/FixedMinimalSmallModelChekpoint/2/model1299.model",
                  "/users/ak1774/scratch/esport/cluster_results/FixedMediumChekpoint/1/model599.model",
                  "/mnt/lustre/users/ak1774/esport/cluster_results/FixedAllChekpoint/0/model599.model"
  
                    ]
configPathList = ["/users/ak1774/scratch/esport/cluster_results/FixedMinimal/3/config.json",
                  "/users/ak1774/scratch/esport/cluster_results/FixedMinimalSmallModelChekpoint/2/config.json",
                  "/users/ak1774/scratch/esport/cluster_results/FixedMediumChekpoint/1/config.json",
                  "/users/ak1774/scratch/esport/cluster_results/FixedAllChekpoint/0/config.json"
                ]

DEBUG_MODE = False
if DEBUG_MODE == False:
    WORKER_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    NUM_WORKERS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

else:
    WORKER_ID = 0
    NUM_WORKERS = 10




num_matches = len(dataPathList)


match_per_worker = int(math.ceil(float(num_matches) / NUM_WORKERS))
print("Match per worker: ",match_per_worker)
sys.stdout.flush()

first_match_index_for_this_task = match_per_worker * WORKER_ID
print("My first match is  ",first_match_index_for_this_task, " num matches is ",num_matches)
sys.stdout.flush()


all_y = [ [] for model_path in modelPathList]
all_pred = [ [] for model_path in modelPathList]

per_sec_pred = [ [[] for _ in range(20)] for model_path in modelPathList ]



for i in range(match_per_worker):
    match_index = first_match_index_for_this_task + i
    if match_index >= num_matches:
        continue
        
    print("Loading match ",match_index)

    data = data_loader.load_data_from_file(dataPathList[match_index])
    
    # get death times
    labels = [(i,label) for i,label in  enumerate(list(data))]
    death_time_indicies = preprocess.labels_to_indicies(preprocess.select_features_by_name("time_until_next_death",labels))
    death_times = data.values[:,death_time_indicies].astype(np.float32)


    for model_i,(model_path,config_path) in enumerate(zip(modelPathList,configPathList)):
    
        with open(config_path) as f:
            config = commentjson.load(f)

        modeldata = test_model.load_pytorch_model(model_path,config,data)

        with torch.no_grad():
            y = modeldata.fullGameLabels
            X = [torch.from_numpy(hero_X) for hero_X in modeldata.fullGameData]
            pred = modeldata.model(X)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()

            all_y[model_i].append(y)
            all_pred[model_i].append(pred)

            for timeslot_i in range(19):
                mask_die_in_timeslot = np.logical_and( (death_times > timeslot_i), (death_times < (timeslot_i+1)))
                per_sec_pred[model_i][timeslot_i].extend(pred[mask_die_in_timeslot].reshape(-1))
            
            mask_die_in_timeslot = (death_times > 19)
            per_sec_pred[model_i][19].extend(pred[mask_die_in_timeslot].reshape(-1))


RESULT_PATH = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_pred_results/test_results/"

print("Predicting done, saving...")

with open(RESULT_PATH + "all_y_" + str(WORKER_ID) +  ".pickle", 'wb') as f:
    pickle.dump(all_y, f, pickle.HIGHEST_PROTOCOL)

with open(RESULT_PATH + "all_pred_" + str(WORKER_ID) +  ".pickle", 'wb') as f:
    pickle.dump(all_pred, f, pickle.HIGHEST_PROTOCOL)

with open(RESULT_PATH + "per_sec_pred_" + str(WORKER_ID) +  ".pickle", 'wb') as f:
    pickle.dump(per_sec_pred, f, pickle.HIGHEST_PROTOCOL)

if WORKER_ID == 0:
    with open(RESULT_PATH + "model_path_list.pickle", 'wb') as f:
        pickle.dump(modelPathList, f, pickle.HIGHEST_PROTOCOL)

print("Saving done")

