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

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

class DotaDataset(Dataset):
    def __init__(self,file_list,batch_size,epoch_size,feature_indicies,label_indicies,who_dies_next_mode,is_validation):
        self.file_list = file_list
        self.feature_indicies = feature_indicies
        self.label_indicies = label_indicies
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.who_dies_next_mode = who_dies_next_mode
        self.is_validation = is_validation
        
    def __getitem__(self, index):  # because how we design our data loadin, this returns a whole batch
        data = data_loader.load_data_from_file(np.random.choice(np.array(self.file_list)))# random file in here
        if self.who_dies_next_mode == True or self.is_validation == True:
            X,y,death_times = data_loader.getBatchBalanced(data,self.batch_size,self.feature_indicies,self.label_indicies,get_death_times=True)
            return X,y,death_times
        else:
            player_i = np.random.randint(10)
            X,y,death_times = data_loader.getBalancedBatchForPlayer(data,player_i,self.batch_size,self.feature_indicies,self.label_indicies,get_death_times=True)
            return X,y,death_times,player_i

    def __len__(self):  
        return self.epoch_size  



def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)




    # is there a config in the current directory?

def get_config():
    config_path = "config.json"
    if not os.path.isfile("config.json"):
        # use default config
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/config/default.json"

    with open(config_path) as f:
        config = commentjson.load(f)
    return config

def load_pytorch_model(modelPath,hero_feature_indicies,label_indicies,config):
    model_type = locate(config["model"])

    inputFeatureSize = len(hero_feature_indicies[0])
    print('input feature size ' + str(inputFeatureSize))
    outputFeatureSize = len(label_indicies)


    model = model_type(inputFeatureSize,outputFeatureSize,**config["model_params"])
    #model.to(device)
    print(model.final_layers)

    print(model.parameters().__next__())

    
    #optimizer = OptimizerType(model.parameters(), **config["optimizer_params"])  


    if config["optimizer"] == "Adam":
        OptimizerType = torch.optim.Adam
    elif config["optimizer"] == "SGD":
        OptimizerType = torch.optim.SGD

    optimizer = OptimizerType

    #checkpoint = torch.load(modelPath)
    model.load_state_dict(torch.load('model.model'))
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    #print(loss)

    return model


heroStuff = []
labelStuff = []
def get_average(l,maxWindow,windowSize,timeStep,label):
    
    #print(len(np.array(l[::-1][0:windowSize])))
    data = np.array(l[::-1][0:windowSize]).sum(axis=0) / windowSize
    #heroStuff.append(data)
    heroStuff.append(l[len(l)-1])
    labelStuff.append(np.squeeze(label,0))

    avg = np.argmax(np.array(l[::-1][0:windowSize]).sum(axis=0) / windowSize)
    if avg >0.1:
        print(
            colored(str(timeStep).zfill(5) + '-  Window size : ' + str(windowSize).zfill(2)  + ' ---- ' + str(avg) + '  ' +str(np.argmax(label)),'red')
            )
        print('----------------------')
    #else:
    #       print('Window size : ' + str(windowSize).zfill(2)  + ' ---- ' + str(avg))

#def plot():


def split_chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def make_predictions(file,modelPath):
    trainingDataFiles = ['data.h5']#glob.glob("/scratch/staff/ak1774/shared_folder/data/train/*.h5")
   
    config = get_config()
    get_feature_indicies_fn = locate(config["feature_set"])
    get_label_indicies_fn = locate(config["lable_set"])

    example_data = data_loader.load_data_from_file(trainingDataFiles[0])
    #hero_feature_indicies,label_indicies = get_feature_indicies_fn(example_data)
    hero_feature_indicies = get_feature_indicies_fn(example_data)
    label_indicies = get_label_indicies_fn(example_data)

    print(len(hero_feature_indicies))
    print(len(label_indicies))
    model = load_pytorch_model(modelPath,hero_feature_indicies,label_indicies,config)
    #model = model.eval()

    print(model)
    data = data_loader.load_data_from_file(file)

    xLims = data['time'].values

    fullGameData,fullGameLabels = data_loader.getSequencialNaive(data,hero_feature_indicies,label_indicies)


    print(np.array(fullGameData).shape)


    windowData = []
    maxWindow = 20


    #torch.set_printoptions(precision=10)
    currentMeanAccuracy = 0
    for i in range(0,19326):

        y = fullGameLabels[i]
        y = np.array(y)
        y = np.expand_dims(y,0)

        X = [torch.from_numpy(hero_X[i:(i+1),:]) for hero_X in fullGameData]

        predX = model(X)
        predX = torch.sigmoid(predX)

        predX = predX.cpu().detach().numpy()

        accuracy_vec = ((predX > 0.5) == (y > 0.5)).reshape(-1).astype(np.float32)

        currentMeanAccuracy +=np.mean( accuracy_vec)

        prediction = predX
        prediction = np.squeeze(prediction,0)
        #print(prediction)
        #print(prediction.shape)
        if i %3000 ==0:
            print('Current mean ' +str(currentMeanAccuracy/(i+1)))


        #print('----------------------')
        windowData.append(prediction)
        print(np.array(windowData).shape)
        
        if len(windowData) > maxWindow:
            windowData.pop(0)

        
        #(get_average(windowData,maxWindow,5,i,y))
        #(get_average(windowData,maxWindow,10,i,y))
        (get_average(windowData,maxWindow,15,i,y))
        #(get_average(windowData,maxWindow,20,i,y))

    heroStuff1 = np.swapaxes(heroStuff,0,1)
    labelStuff1= np.swapaxes(labelStuff,0,1)

    #heroStuff1 = heroStuff
    #labelStuff1 = labelStuff
    #x = arange(0,len(heroStuff1[0]))
    #x = np.array(x) /
    splitLower = 2500
    splitHigher = 3000

    xLims = xLims - xLims[0] - 90
    
    print(np.array(heroStuff1).shape)        
    print(np.array(labelStuff1).shape)
    print(np.array(xLims).shape)

    '''
    heroStuff1 = heroStuff1[:,splitLower:splitHigher]
    labelStuff1 = labelStuff1[:,splitLower:splitHigher]
    xLims = xLims[splitLower:splitHigher]
    '''


    #print(heroStuff1[0])
    heroStuff1 = (heroStuff1 -1)
    labelStuff1 = (labelStuff1 -1) * -1
    #print(heroStuff1[0])
    #heroStuff1[:] = [x - 1 for x in heroStuff1]
    #labelStuff1[:] = [(x - 1) * -1 for x in labelStuff1]


    # 1:30 game start
    

    print(np.array(heroStuff1).shape)        
    print(np.array(labelStuff1).shape)
    print(np.array(xLims).shape)
    #x = np.arange(194) 
    #plt.subplots_adjust(hspace=100)
    #plt.xticks(np.arange(0,1,))

    #fig = plt.figure(figsize=(11,8))
    #ax1 = fig.add_subplot(111)
    #plt.yticks(np.arange(0, 1, step=1))
    for i in range(0,10):
        #ax1 = fig.add_subplot(111)
        #ax1.plot(heroStuff1[i], label=1)
        #ax1.plot(labelStuff1[i], label=2)
        plt.subplot(10, 1, (i+1))
        plt.plot(xLims, heroStuff1[i] ,color='red',linewidth=0.5)
        plt.plot(xLims,labelStuff1[i] ,color='blue',linewidth=0.5)
        #plt.title('Player ' + str(i))

    #ax1.legend(loc=2)

    plt.savefig('smooth_plot.eps')



def calculate_detailed_accuracies(accuracy_vec,death_times,y):
    
    overall_accuracy = accuracy_vec.mean()

    no_kill_accuracy = []
    kill_accuracy = []

    one_sec_accuracy = []
    two_sec_accuracy = []
    three_sec_accuracy = []
    four_sec_accuracy = []
    five_sec_accuracy = []

    for batch_i,accuracy in enumerate(accuracy_vec):
        if y[batch_i] == 10:
            no_kill_accuracy.append(accuracy)
        else:
            kill_accuracy.append(accuracy)

            if death_times[batch_i,y[batch_i]] < 1:
                one_sec_accuracy.append(accuracy)
            elif death_times[batch_i,y[batch_i]] < 2:
                two_sec_accuracy.append(accuracy)
            elif death_times[batch_i,y[batch_i]] < 3:
                three_sec_accuracy.append(accuracy)
            elif death_times[batch_i,y[batch_i]] < 4:
                four_sec_accuracy.append(accuracy)
            else:
                five_sec_accuracy.append(accuracy)

    return (overall_accuracy,(kill_accuracy,no_kill_accuracy),
            (one_sec_accuracy,two_sec_accuracy,three_sec_accuracy,four_sec_accuracy,five_sec_accuracy))





def train_pytorch():
    
    # is there a config in the current directory?
    config_path = "config.json"
    if not os.path.isfile("config.json"):
        # use default config
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/config/default.json"

    with open(config_path) as f:
        config = commentjson.load(f)


    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    sys.stdout.flush()

    WHO_DIES_NEXT_MODE = config["predict_who_dies_next"]

    use_cuda = config["use_gpu"]==True and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("using device: ",device)

    model_type = locate(config["model"])
    get_feature_indicies_fn = locate(config["feature_set"])
    get_label_indicies_fn = locate(config["lable_set"])

    batch_size = config["batch_size"]
    print(type(batch_size))
    print(type(config["log_at_every_x_sample"]))
    epoch_size = int(config["log_at_every_x_sample"] / batch_size)
    print("epoch_size: ", epoch_size)
    checkpoint_frequency = int(config["chekpoint_at_every_x_sample"] / (epoch_size * batch_size))
    validation_epoch_sice = config["validation_epoch_size"]

    if config["optimizer"] == "Adam":
        OptimizerType = torch.optim.Adam
    elif config["optimizer"] == "SGD":
        OptimizerType = torch.optim.SGD


    # YARCC
    #trainingDataFiles = glob.glob("/scratch/ak1774/data/train/*.h5")
    #validationDataFiles = glob.glob("/scratch/ak1774/data/validation/*.h5")

    # Viking
    trainingDataFiles = glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/train/*.h5")
    validationDataFiles = glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/validation/*.h5")

    #trainingDataFiles = glob.glob("/scratch/staff/ak1774/shared_folder/data/train/*.h5")
    #validationDataFiles = glob.glob("/scratch/staff/ak1774/shared_folder/data/validation/*.h5")

    example_data = data_loader.load_data_from_file(trainingDataFiles[0])
    hero_feature_indicies = get_feature_indicies_fn(example_data)
    
    if WHO_DIES_NEXT_MODE == True:
        label_indicies = get_label_indicies_fn(example_data)
    else:
        label_indicies = get_label_indicies_fn(example_data,config["label_set_arg"])

    

    inputFeatureSize = len(hero_feature_indicies[0])
    outputFeatureSize = len(label_indicies)

    if WHO_DIES_NEXT_MODE == True and outputFeatureSize != 11:
        print("error, bad config, label set and prediction mode mismatch")
        raise "error, bad config, label set and prediction mode mismatch"
    elif WHO_DIES_NEXT_MODE == False and outputFeatureSize != 10:
        print("error, bad config, label set and prediction mode mismatch")
        raise "error, bad config, label set and prediction mode mismatch"



    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    training_set = DotaDataset(file_list=trainingDataFiles,batch_size=batch_size,epoch_size=epoch_size,
                              feature_indicies=hero_feature_indicies,label_indicies=label_indicies,who_dies_next_mode=WHO_DIES_NEXT_MODE,is_validation=False) # set is validation to get death times...
    training_generator = torch.utils.data.DataLoader(training_set,num_workers=20,worker_init_fn=worker_init_fn)

    validation_set = DotaDataset(file_list=validationDataFiles,batch_size=batch_size,epoch_size=validation_epoch_sice,
                              feature_indicies=hero_feature_indicies,label_indicies=label_indicies,who_dies_next_mode=WHO_DIES_NEXT_MODE,is_validation=False) # actually we want the same distribution, so we can compare loss, so dont do anything differently in case of validation
    validation_generator = torch.utils.data.DataLoader(validation_set,num_workers=20,worker_init_fn=worker_init_fn)


    #model = models.SimpleFF(inputFeatureSize,outputFeatureSize)
    model = model_type(inputFeatureSize,outputFeatureSize,**config["model_params"])
    model.to(device)
    print(model.final_layers)

    criterion = nn.CrossEntropyLoss()  
    binary_classification_loss = torch.nn.BCELoss()
    optimizer = OptimizerType(model.parameters(), **config["optimizer_params"])  


    if WHO_DIES_NEXT_MODE == True:
        
        all_train_losses = []
        all_train_accuracies = []
        all_train_kill_nokill_accuracies = []
        all_train_per_second_accuracies = []

        all_validation_losses = []
        all_validation_accuracies = []
        all_validation_kill_nokill_accuracies = []
        all_validation_per_second_accuracies = []

        for epoch_i in range(50000):

            now = time.time()

            np.random.seed() # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values

            epoch_losses = []
            epoch_overall_accuracies = []
            epoch_kill_accuracies = []
            epoch_no_kill_accuracies = []
            epoch_one_sec_accuracies = []
            epoch_two_sec_accuracies = []
            epoch_three_sec_accuracies = []
            epoch_four_sec_accuracies = []
            epoch_five_sec_accuracies = []

            for sub_epoch_i,(X,y,death_times) in enumerate(training_generator):
                
                # since we get a batch of size 1 of batch of real batch size, we take the 0th element
                X = [(hero_X[0,:]).to(device) for hero_X in X]
                y = torch.argmax(y[0,:],dim=1).to(device)
                death_times = death_times[0]

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                output = model(X)

                loss = criterion(output, y)
                accuracy_vec = (torch.argmax(output,1) == y).cpu().numpy().reshape(-1).astype(np.float32)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.cpu().detach().numpy().reshape(-1)[0])

                (overall_accuracy,(kill_accuracy,no_kill_accuracy),
                (one_sec_accuracy,two_sec_accuracy,three_sec_accuracy,four_sec_accuracy,five_sec_accuracy)) = calculate_detailed_accuracies(accuracy_vec,death_times,y)

                epoch_overall_accuracies.append(overall_accuracy)
                epoch_kill_accuracies.extend(kill_accuracy)
                epoch_no_kill_accuracies.extend(no_kill_accuracy)
                epoch_one_sec_accuracies.extend(one_sec_accuracy)
                epoch_two_sec_accuracies.extend(two_sec_accuracy)
                epoch_three_sec_accuracies.extend(three_sec_accuracy)
                epoch_four_sec_accuracies.extend(four_sec_accuracy)
                epoch_five_sec_accuracies.extend(five_sec_accuracy)

                if sub_epoch_i > 0 and (sub_epoch_i % 50) == 0:
                    print(epoch_i," ",sub_epoch_i," loss: ",np.array(epoch_losses[-49:]).mean()," accuracy: ",np.array(epoch_overall_accuracies[(-49*y.shape[0]):]).mean())
                    sys.stdout.flush()


            

            all_train_losses.append(np.array(epoch_losses).mean())
            all_train_accuracies.append(np.array(epoch_overall_accuracies).mean())
            all_train_kill_nokill_accuracies.append( (np.array(epoch_kill_accuracies).mean(), np.array(epoch_no_kill_accuracies).mean()) )
            all_train_per_second_accuracies.append( (
                                            np.array(epoch_one_sec_accuracies).mean(),
                                            np.array(epoch_two_sec_accuracies).mean(),
                                            np.array(epoch_three_sec_accuracies).mean(),
                                            np.array(epoch_four_sec_accuracies).mean(),
                                            np.array(epoch_five_sec_accuracies).mean()
                                            ))



    

            # reset logs for validation
            epoch_losses = []
            epoch_overall_accuracies = []
            epoch_kill_accuracies = []
            epoch_no_kill_accuracies = []
            epoch_one_sec_accuracies = []
            epoch_two_sec_accuracies = []
            epoch_three_sec_accuracies = []
            epoch_four_sec_accuracies = []
            epoch_five_sec_accuracies = []

            with torch.no_grad():
                for X,y,death_times in validation_generator:
                    X = [(hero_X[0,:]).to(device) for hero_X in X]
                    y = torch.argmax(y[0,:],dim=1).to(device)
                    death_times = death_times[0]

                    output = model(X)

                    loss = criterion(output, y)
                    accuracy_vec = (torch.argmax(output,1) == y).cpu().numpy().reshape(-1).astype(np.float32)

                    epoch_losses.append(loss.cpu().detach().numpy().reshape(-1)[0])

                    (overall_accuracy,(kill_accuracy,no_kill_accuracy),
                    (one_sec_accuracy,two_sec_accuracy,three_sec_accuracy,four_sec_accuracy,five_sec_accuracy)) = calculate_detailed_accuracies(accuracy_vec,death_times,y)

                    epoch_overall_accuracies.append(overall_accuracy)
                    epoch_kill_accuracies.extend(kill_accuracy)
                    epoch_no_kill_accuracies.extend(no_kill_accuracy)
                    epoch_one_sec_accuracies.extend(one_sec_accuracy)
                    epoch_two_sec_accuracies.extend(two_sec_accuracy)
                    epoch_three_sec_accuracies.extend(three_sec_accuracy)
                    epoch_four_sec_accuracies.extend(four_sec_accuracy)
                    epoch_five_sec_accuracies.extend(five_sec_accuracy)

                        
            all_validation_losses.append(np.array(epoch_losses).mean())
            all_validation_accuracies.append(np.array(epoch_overall_accuracies).mean())
            all_validation_kill_nokill_accuracies.append( (np.array(epoch_kill_accuracies).mean(), np.array(epoch_no_kill_accuracies).mean()) )
            all_validation_per_second_accuracies.append( (
                                            np.array(epoch_one_sec_accuracies).mean(),
                                            np.array(epoch_two_sec_accuracies).mean(),
                                            np.array(epoch_three_sec_accuracies).mean(),
                                            np.array(epoch_four_sec_accuracies).mean(),
                                            np.array(epoch_five_sec_accuracies).mean()
                                            ))

            # epoch over, checkpoint, report, check validation error
            print("Epoch done ",epoch_i," loss: ",np.array(epoch_losses).mean()," accuracy: ",np.array(epoch_overall_accuracies).mean())



            #print("all_train_kill_nokill_accuracies ",len(all_train_kill_nokill_accuracies))
            PlotValues((all_train_losses,all_validation_losses),"loss",["train","validation"])
            PlotValues((all_train_accuracies,all_validation_accuracies),"accuracy",["train","validation"])


            PlotValues((*zip(*all_train_kill_nokill_accuracies),*zip(*all_validation_kill_nokill_accuracies)),"accuracy_kill",
                            ["train_kill","train_no_kill","validation_kill","validation_no_kill"])

            sec_labels = ["1_sec","2_sec","3_sec","4_sec","5_sec"]
            PlotValues((*zip(*all_train_per_second_accuracies),*zip(*all_validation_per_second_accuracies))  ,"accuracy_sec",
                    [*[ "accuracy_train" + label for label in sec_labels], *[ "accuracy_validation" + label for label in sec_labels]])


            #np.save('losses.npy', np.array(mean_losses))
            #np.save('accuracies.npy', np.array(mean_accuracies))

            print("Epoch took: ",time.time()-now)
            sys.stdout.flush()

            #PlotValues(mean_validation_accuracies,"valid_accuracy")
            #PlotValues(mean_valid_overall_accuracies,"valid_overall_accuracy")

            #np.save('mean_valid_overall_accuracies.npy', np.array(mean_valid_overall_accuracies))
            #np.save('mean_validation_accuracies.npy', np.array(mean_validation_accuracies))

            if (epoch_i % 100) == 99:
                torch.save(model.state_dict(), "model" + str(epoch_i) + ".model")
    
    
    else:  # Per player probability prediction


        all_train_losses = []
        all_train_accuracies = []
        all_train_target_accuracies = []
        all_train_die_notdie_accuracies = []
        all_train_per_sec_accuracies = [[] for _ in range(20)]
        all_train_per_sec_predictions = [[] for _ in range(20)]
        all_train_per_sec_predictions_std = [[] for _ in range(20)]

        all_validation_losses = []
        all_validation_accuracies = []

        all_validation_roc_scores = []
        all_validation_pr_scores = []


        for epoch_i in range(50000):

            now = time.time()

            np.random.seed() # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values

            epoch_overall_loss = []
            epoch_overall_accuracy = []
            epoch_target_accuracy = []
            epoch_die_accuracy = []
            epoch_not_die_accuracy = []
            epoch_per_sec_accuracies = [[] for _ in range(20)]
            epoch_per_sec_predictions = [[] for _ in range(20)]

            for sub_epoch_i,(X,y,death_times,player_i) in enumerate(training_generator):
                # since we get a batch of size 1 of batch of real batch size, we take the 0th element
                X = [(hero_X[0,:]).to(device) for hero_X in X]
                y = (y[0,:]).to(device)
                death_times = death_times[0]
                player_i = player_i[0].to(device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                output = model(X)
                output = torch.sigmoid(output)
                output_np = output.cpu().detach().numpy()

                # only backpropagate the loss for player_i (so the training data is balanced)
                loss = binary_classification_loss(output[:,player_i], y[:,player_i])

                loss.backward()
                optimizer.step()

                overall_loss = binary_classification_loss(output, y).cpu().detach().numpy()
                epoch_overall_loss.append(overall_loss.reshape(-1).mean())
                accuracy_values = ((output > 0.5) == (y > 0.5)).cpu().numpy().astype(np.float32)

                target_accuracy = ((output[:,player_i] > 0.5) == (y[:,player_i] > 0.5)).cpu().numpy().reshape(-1).astype(np.float32)

                die_accuracy_vec = ((output > 0.5) == (y > 0.5)).view(-1)[y.view(-1) > 0.5].cpu().numpy().reshape(-1).astype(np.float32)
                not_die_accuracy_vec = ((output > 0.5) == (y > 0.5)).view(-1)[y.view(-1) < 0.5].cpu().numpy().reshape(-1).astype(np.float32)

                epoch_overall_accuracy.append(accuracy_values.reshape(-1).mean())
                epoch_target_accuracy.append(target_accuracy.mean())

                # these have varying size, so calculating the proper mean across batches takes more work
                epoch_die_accuracy.extend(die_accuracy_vec)
                epoch_not_die_accuracy.extend(not_die_accuracy_vec)
                

                death_times = death_times.cpu().numpy()
                #death_times[death_times < 0] = 1000.0 # make invalid death times a large number


                for timeslot_i in range(19):
                    mask_die_in_timeslot = np.logical_and( (death_times > timeslot_i), (death_times < (timeslot_i+1)))
                    epoch_per_sec_accuracies[timeslot_i].extend(accuracy_values[mask_die_in_timeslot].reshape(-1))
                    epoch_per_sec_predictions[timeslot_i].extend(output_np[mask_die_in_timeslot].reshape(-1))

                # and the rest
                mask_die_in_timeslot = (death_times > 19)
                epoch_per_sec_accuracies[19].extend(accuracy_values[mask_die_in_timeslot].reshape(-1))
                epoch_per_sec_predictions[19].extend(output_np[mask_die_in_timeslot].reshape(-1))


                if sub_epoch_i > 0 and (sub_epoch_i % 50) == 0:
                    print(epoch_i," ",sub_epoch_i," loss: ",np.array(epoch_overall_loss[-49:]).mean()," accuracy: ",np.array(epoch_target_accuracy[-49:]).mean())
                    #for timeslot_i in range(19):
                    #    print("epoch_per_sec_predictions  ",len(epoch_per_sec_predictions[timeslot_i]))
                    
                    #print("die accuracy: ",np.array(epoch_die_accuracy[-49:]).mean())
                    #print("not_die accuracy: ",np.array(epoch_not_die_accuracy[-49:]).mean())
                    sys.stdout.flush()


            if (epoch_i % 10) == 9:
                np.save('epoch_per_sec_predictions.npy', np.array(epoch_per_sec_predictions))

            all_train_losses.append(np.array(epoch_overall_loss).mean())
            all_train_accuracies.append(np.array(epoch_overall_accuracy).mean())
            all_train_target_accuracies.append(np.array(epoch_target_accuracy).mean())
            all_train_die_notdie_accuracies.append((np.array(die_accuracy_vec).mean(),np.array(not_die_accuracy_vec).mean()))

            for timeslot_i in range(20):
                all_train_per_sec_accuracies[timeslot_i].append(np.array(epoch_per_sec_accuracies[timeslot_i]).mean())
                all_train_per_sec_predictions[timeslot_i].append(np.array(epoch_per_sec_predictions[timeslot_i]).mean())
                all_train_per_sec_predictions_std[timeslot_i].append(np.array(epoch_per_sec_predictions[timeslot_i]).std())


            # VALIDATION EPOCH
            if (epoch_i % 3) == 0:

                epoch_overall_loss = []
                epoch_overall_accuracy = []

                epoch_all_pred = []
                epoch_all_y = []
                #epoch_die_accuracy = []
                #epoch_not_die_accuracy = []
                #epoch_per_sec_accuracies = [[] for _ in range(20)]
                #epoch_per_sec_predictions = [[] for _ in range(20)]

                with torch.no_grad():
                    for X,y,death_times,player_i in validation_generator:


                        X = [(hero_X[0,:]).to(device) for hero_X in X]
                        y = (y[0,:]).to(device)
                        death_times = death_times[0]

                        output = model(X)
                        output = torch.sigmoid(output)
                        output_np = output.cpu().detach().numpy()

                        epoch_overall_loss.append(binary_classification_loss(output, y).cpu().detach().numpy().reshape(-1).mean())
                        accuracy_vec = ((output > 0.5) == (y > 0.5)).cpu().numpy().reshape(-1).astype(np.float32)
                        epoch_overall_accuracy.append(accuracy_vec.mean())

                        #death_times = death_times.cpu().numpy()

                         #for timeslot_i in range(19):
                         #   mask_die_in_timeslot = np.logical_and( (death_times > timeslot_i), (death_times < (timeslot_i+1)))
                         #   epoch_per_sec_accuracies[timeslot_i].extend(accuracy_values[mask_die_in_timeslot].reshape(-1))
                         #   epoch_per_sec_predictions[timeslot_i].extend(output_np[mask_die_in_timeslot].reshape(-1))

                        epoch_all_pred.extend(output_np.reshape(-1))
                        epoch_all_y.extend(y.cpu().numpy().reshape(-1))


                all_validation_roc_scores.append(roc_auc_score(epoch_all_y, epoch_all_pred))
                all_validation_pr_scores.append(average_precision_score(epoch_all_y, epoch_all_pred))


                all_validation_losses.append(np.array(epoch_overall_loss).mean())
                all_validation_accuracies.append(np.array(epoch_overall_accuracy).mean())
            else:
                # just copy the previous validation statistics, so we can plot it togeather with training statistics
                all_validation_losses.append(all_validation_losses[-1])
                all_validation_accuracies.append(all_validation_accuracies[-1])
                all_validation_roc_scores.append(all_validation_roc_scores[-1])
                all_validation_pr_scores.append(all_validation_pr_scores[-1])
           
            PlotValues((all_train_losses,all_validation_losses),"loss",["train","validation"])
            PlotValues((all_train_accuracies,all_validation_accuracies),"accuracy",["train","validation"])

            PlotValues((all_validation_roc_scores,),"roc_score",["roc"])
            PlotValues((all_validation_pr_scores,),"pr_score",["pr"])

            #PlotValues((all_train_losses,),"loss",["train"])
            #PlotValues((all_train_accuracies,),"accuracy",["train"])
            PlotValues((all_train_target_accuracies,),"target_accuracy",["train"])

            PlotValues( ([vals[0] for vals in all_train_die_notdie_accuracies],
                         [vals[1] for vals in all_train_die_notdie_accuracies]),"all_train_die_notdie_accuracies",["die","not_die"])

            PlotValues(all_train_per_sec_accuracies,"all_train_per_sec_accuracies",[ str(time_i+1)+"_sec" for time_i in range(20) ])

            PlotWithStd(values = [vec[-1] for vec in all_train_per_sec_predictions],
                        stds =   [vec[-1] for vec in all_train_per_sec_predictions_std],
                        legends = ["per_sec predictions"],
                        name = "per_sec predictions")



            print("Epoch done ",epoch_i," loss: ",np.array(epoch_overall_loss).mean()," accuracy: ",np.array(epoch_target_accuracy).mean())
            print("Epoch took: ",time.time()-now)
            sys.stdout.flush()

            if (epoch_i % 10) == 9:
                np.save('all_train_per_sec_predictions.npy', np.array(all_train_per_sec_predictions))
                np.save('all_train_per_sec_predictions_std.npy', np.array(all_train_per_sec_predictions_std))
                

            if (epoch_i % 100) == 99:
                torch.save(model.state_dict(), "model" + str(epoch_i) + ".model")

#Plot Train/Test acccuracy
def PlotValues(values,name,legends):
    plt.clf()
    for vec in values:
        plt.plot(vec)
    
    plt.ylabel(name)		
    plt.xlabel('Epoch')

    plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(name,bbox_inches='tight', dpi=300)
		
def PlotWithStd(values,stds,legends,name):
    plt.clf()

    plt.errorbar(list(range(len(values))),values,yerr=stds,fmt='-o')

    plt.ylabel(name)		
    plt.xlabel('Epoch')

    plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(name,bbox_inches='tight', dpi=300)






if __name__ == "__main__":

    #make_predictions('3475430774_414202435_whole_match_normalized.h5','model.model')
    train_pytorch()









