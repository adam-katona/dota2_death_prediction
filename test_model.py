import os
import random
import glob
import itertools
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import matplotlib.animation as animation

# in case it is called from a different location
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import data_loader
import models

from termcolor import colored

import commentjson
from pydoc import locate

import pickle



    # is there a config in the current directory?

def get_config(config_path="config.json"):
    with open('ModelData/' + config_path) as f:
        config = commentjson.load(f)
    return config


class ModelData:
    def __init__(self, model, fullGameData,fullGameLabels ):
        self.model = model
        self.fullGameData = fullGameData
        self.fullGameLabels = fullGameLabels


def load_pytorch_model(modelPath,config,data):


    get_feature_indicies_fn = locate(config["feature_set"])
    get_label_indicies_fn = locate(config["lable_set"])

    
    hero_feature_indicies = get_feature_indicies_fn(data)
    label_indicies = get_label_indicies_fn(data,config["label_set_arg"])


    model_type = locate(config["model"])
    inputFeatureSize = len(hero_feature_indicies[0])
    outputFeatureSize = len(label_indicies)
    model = model_type(inputFeatureSize,outputFeatureSize,**config["model_params"])

    if config["optimizer"] == "Adam":
        OptimizerType = torch.optim.Adam
    elif config["optimizer"] == "SGD":
        OptimizerType = torch.optim.SGD

    optimizer = OptimizerType

    model.load_state_dict(torch.load(modelPath))
    print('loaded model')

    fullGameData,fullGameLabels = data_loader.getSequencialNaive(data,hero_feature_indicies,label_indicies)

    return ModelData(model,fullGameData,fullGameLabels) 



heroStuff = []
heroStuffWindow = []
labelStuff = []

def modelPred(model,X):
    predX = model(X)
    predX = torch.sigmoid(predX)
    predX = predX.cpu().detach().numpy()
    return predX

def averagePred(models,X):
    vals =[]
    for m in models:
        vals = modelPred(m,X)+ vals

    return (vals / len(models))


#def dump_predictions(file,modelPath)


def make_predictions(file,modelPath):
    trainingDataFiles = [file]#glob.glob("/scratch/staff/ak1774/shared_folder/data/train/*.h5")
    data = data_loader.load_data_from_file(trainingDataFiles[0])

    models = []
    for i in range(1,2):
        print(i)
        models.append( load_pytorch_model('ModelData/' +str(i) +'/' +'model.model',
                            get_config('/' +str(i) +'/config.json'), data) )



    


    #fullGameData,fullGameLabels = data_loader.getSequencialNaive(data,hero_feature_indicies,label_indicies)


    xLims = data['time'].values

    #¢health = data['player_4_m_iHealth'].values

    
    #######################
    # get original health
    ######################

    norm_stats = None
    with open("norm_stats.pickle", 'rb') as f:
        norm_stats = pickle.load(f)

    for label,min_value,max_value in normalization_stats:
        if "_m_iHealth" in label:
            health_min = min_value
            health_max = max_value
        if "m_iMaxHealth" in label:
            maxhealth_min = min_value
            maxhealth_max = max_value

    healthes = []
    max_healthes = []
    relative_healthes = []
    for i in range(0,10):
        health_vals = data['player_' + str(i) + '_m_iHealth'].values
        maxhealth_vals = data['player_' + str(i) + '_m_iMaxHealth'].values

        health_vals = health_vals * (health_max - health_min) + health_min
        maxhealth_vals = maxhealth_vals * (maxhealth_max - maxhealth_min) + maxhealth_min

        relative_health_vals = health_vals / maxhealth_vals # hopefully maxhealth is never 0

        healthes.append(health_vals)
        max_healthes.append(maxhealth_vals)
        relative_healthes.append(relative_health_vals)


    #######################
    # get death times
    ######################

    labels = [(i,label) for i,label in  enumerate(list(data))]
    death_time_indicies = preprocess.labels_to_indicies(preprocess.select_features_by_name("time_until_next_death",labels))
    death_times = data.values[:,death_time_indicies].astype(np.float32)



    for m in models:

        X = [torch.from_numpy(hero_X) for hero_X in m.fullGameData]

        pred = model(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach().numpy()

        y = m.fullGameLabels




    currentMeanTrueAccuracy = 0
    currentMeanFalseAccuracy=0

    numTruePos = 0
    numFalsePos = 0
    numTrueNeg = 0
    numFalseNeg = 0
    for i in range(0,data.shape[0]):
        predX = 0
        for m in models:
            y = m.fullGameLabels[i]
            y = np.array(y)
            y = np.expand_dims(y,0)

            X = [torch.from_numpy(hero_X[i:(i+1),:]) for hero_X in m.fullGameData]
            print(i)
            #predX = averagePred(models,X)
            predX = modelPred(m.model,X) +predX


        predX = predX/len(models)

        '''
        true_pos = ((predX > 0.5) == (y > 0.5)).reshape(-1).astype(np.float32)
        true_neg = ((predX < 0.5) == (y <0.5)).reshape(-1).astype(np.float32)
        false_pos = ((predX > 0.5) == (y < 0.5)).reshape(-1).astype(np.float32)
        false_neg = ((predX < 0.5) == (y > 0.5)).reshape(-1).astype(np.float32)

        for pos in true_neg:
            if pos ==1:
                numTrueNeg +=1
        for pos in false_neg:
            if pos ==1:
                numFalseNeg +=1

        for pos in true_pos:
            if pos ==1:
                numTruePos +=1
        for pos in false_pos:
            if pos ==1:
                numFalsePos +=1
        '''


        prediction = predX
        currentMeanTrueAccuracy += np.mean(true_pos)
        currentMeanFalseAccuracy += np.mean(false_pos)

        prediction = np.squeeze(prediction,0)
        if i %3000 ==0:
            print('Current true pos ' +str(currentMeanTrueAccuracy/(i+1)))
            print('Current false pos ' +str(currentMeanFalseAccuracy/(i+1)))

        heroStuff.append(prediction)
        labelStuff.append(np.squeeze(y,0))

    print()
    print(numTruePos)
    print(numTrueNeg)
    print()
    print(numFalsePos)
    print(numFalseNeg)

    print()
    print('True Pos = ' + str(currentMeanTrueAccuracy/19326))
    print('False pos = ' + str(currentMeanFalseAccuracy/19326))

    heroStuff1 = np.swapaxes(heroStuff,0,1)
    labelStuff1= np.swapaxes(labelStuff,0,1)

    xLims = xLims - xLims[0] - 90

    np.save('hero.npy', np.array(heroStuff1))
    np.save('label.npy', np.array(labelStuff1))
    np.save('xLims.npy', np.array(xLims))
    np.save('health.npy',np.array(healthes))
  


if __name__ == "__main__":

    make_predictions('3475430774_414202435_whole_match_normalized.h5','model.model')
    #train_pytorch()









