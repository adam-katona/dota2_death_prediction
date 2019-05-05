


import sys
import numpy as np
import math
import copy
import random
import time

import matplotlib.pyplot as plt
import logging
#from colorlog import ColoredFormatter
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output 
#LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

import os
import signal
import datetime
import subprocess

import commentjson
from pydoc import locate
import shutil



EXPERIMENT_NAME = "FixedReallyAllChekpoint"

ENVIRONMENT = "slurm"
#ENVIRONMENT = "slurm_debug"
#ENVIRONMENT = "sge"
#ENVIRONMENT = "sge_debug"
#ENVIRONMENT = "csgpu"
#ENVIRONMENT = "debug"

DEFAULT_CONFIG_PATH = "/users/ak1774/scratch/esport/death_prediction/config/default.json"
RESULTS_PATH = "/users/ak1774/scratch/esport/cluster_results"
TRAINING_SCRIPT_FILE = "/users/ak1774/scratch/esport/death_prediction/train.py"

if ENVIRONMENT == "sge" or ENVIRONMENT == "debug":
    DEFAULT_CONFIG_PATH = "/scratch/ak1774/DeathPrediction/death_prediction/config/default.json"
    RESULTS_PATH = "/scratch/ak1774/DeathPrediction/results"
    TRAINING_SCRIPT_FILE = "/scratch/ak1774/DeathPrediction/death_prediction/train.py"




def generate_job_file_sge(python_script_path):
    lines = []
    lines.append("#!/bin/bash")
    lines.append("")
    lines.append("#$ -cwd -V")
    lines.append("#$ -l h_rt=30:00:00")
    lines.append("#$ -l h_vmem=2G")
    lines.append("#$ -pe smp 8")
    #lines.append("#$ -q iggi-cluster")
    lines.append("")
    lines.append("")
    lines.append("echo `pwd`")
    #lines.append("OMP_NUM_THREADS=1 python {0}".format(python_script_path))
    lines.append("python {0}".format(python_script_path))


    return lines

def generate_job_file(python_script_path):

    lines = []
    lines.append("#!/bin/bash")
    lines.append("#SBATCH --job-name=dota")
    lines.append("#SBATCH --ntasks=1")
    lines.append("#SBATCH --mem=30gb")
    lines.append("#SBATCH --cpus-per-task=10")
    #lines.append("#SBATCH --nodes=1")
    lines.append("#SBATCH --time=120:00:00")
    lines.append("#SBATCH --output=basic_job_%j.log")
    lines.append("#SBATCH --account=CS-DCLABS-2019")
    lines.append("#SBATCH --export=ALL")
#    lines.append("#SBATCH --partition=month ")
#    lines.append("#SBATCH --partition=himem ")
#    lines.append("#SBATCH --partition=gpu ")
#    lines.append("#SBATCH --gres=gpu:1")
    lines.append("")
    lines.append("")
    lines.append("echo `pwd`")
    lines.append("OMP_NUM_THREADS=1 python {0}".format(python_script_path))

    return lines


def generate_random_config(default_config,n):

    configs = []
    for i in range(n):

        new_conf = copy.copy(default_config)

        # choose a feature set
        new_conf["feature_set"] = np.random.choice(
                                [#   "data_loader.getFeatureCorrectMinimal",
                                 #   "data_loader.getFeatureCorrectMedium",
                                    "data_loader.getFeatureCorrectAll"
                                ])

                                   # ["data_loader.getFeatureIndiciesMinimal",
                                   # "data_loader.getFeatureIndiciesSmall",
                                   # "data_loader.getFeatureIndiciesNoAbilityNoHeroNoItems",
                                   #  "data_loader.getFeatureIndiciesNoAbilityNoHero",
                                   #  "data_loader.getFeatureIndiciesAll"])



    


        new_conf["lable_set"] = "data_loader.getLabelIndicies_die_in_n"  
        new_conf["label_set_arg"] =  np.random.choice(
                                    ["die_in_5",
                                 #    "die_in_7.5",
                                 #    "die_in_10",
                                  #   "die_in_12.5",
                                  #   "die_in_15",
                                   #  "die_in_17.5",
                                 #    "die_in_20"
                                     ])


        # choose a model
        new_conf["model"] = "models.SharedHeroWeightsFF"
        new_conf["model_params"] = np.random.choice(
                                    [
                                        #{"shared_layer_sizes" : [200,100,60,20],"final_layer_sizes" : [150,75]}, # 4,2
                                        #{"shared_layer_sizes" : [400,200,120,40],"final_layer_sizes" : [300,150,75,32]}, # 4,4
                                        #{"shared_layer_sizes" : [1024,512,256,128,64],"final_layer_sizes" : [512,256,128,64,32]}, # 5,5
                                        #{"shared_layer_sizes" : [1400,700,350,175,85],"final_layer_sizes" : [512,256,128,64,32]}, # 5,5
                                        #{"shared_layer_sizes" : [2000,1000,500,250,125,62],"final_layer_sizes" : [512,256,128,64,32]}, # 5,5
                                        #{"shared_layer_sizes" : [1024,512,256,128,64,32],"final_layer_sizes" : [150,75]}, # 6,2
                                        {"shared_layer_sizes" : [256,128,64],"final_layer_sizes" : [1024,512,256,128,64,32]}, # 3,6
                                        #{"shared_layer_sizes" : [256,128,64],"final_layer_sizes" : [2000,1000,500,250,100,50]}, # 3,6
                                        #{"shared_layer_sizes" : [400,200,80],"final_layer_sizes" : [1000,500,250,100,50]} # 3,5

                                        #'''
                                        #{"shared_layer_sizes" : [60,20],"final_layer_sizes" : [100]},            # 2,1
                                        #{"shared_layer_sizes" : [100,60,20],"final_layer_sizes" : [100]},        # 3,1
                                        #{"shared_layer_sizes" : [200,100,60,20],"final_layer_sizes" : [100]},    # 4,1
                                        #{"shared_layer_sizes" : [100,60,20],"final_layer_sizes" : [150,75]},     # 3,2
                                        #{"shared_layer_sizes" : [200,100,60,20],"final_layer_sizes" : [150,75]}, # 4,2
                                        #{"shared_layer_sizes" : [200,100,40],"final_layer_sizes" : [300,150,75]} # 3,3
                                        #'''
                                    ])

        # choose a batch size
        new_conf["batch_size"] = int(np.random.choice([128]))

        # choose optimitzer
        new_conf["optimizer"] = "Adam" # np.random.choice(["Adam","SGD"])
        new_conf["optimizer_params"] = { "lr" : 0.0000615  }   # float(10**np.random.uniform(-3.5,-5.0))}
        #if new_conf["optimizer"] == "SGD":
        #    new_conf["optimizer_params"] = { "lr" : 10**np.random.uniform(-1,-5), "nesterov" : np.random.choice(["true","false"]), "momentum": 0.5}
        #else:
        #    

        new_conf["validation_epoch_size"] = 600

        new_conf["use_gpu"] = "true"  # if available

        if ENVIRONMENT == "slurm_debug":
            new_conf["log_at_every_x_sample"] = 7680

        configs.append(new_conf)
    return configs



# Load default config
default_config = None
with open(DEFAULT_CONFIG_PATH) as f:
    default_config = commentjson.load(f)


# Generate configurations
config_list = generate_random_config(default_config,n=10)


# create experiment_dir    
os.makedirs(RESULTS_PATH,exist_ok=True)
#main_experiment_dir = os.path.join(RESULTS_PATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
main_experiment_dir = os.path.join(RESULTS_PATH, EXPERIMENT_NAME)
os.makedirs(main_experiment_dir,exist_ok=True)

os.chdir(main_experiment_dir)

for config_i,config in enumerate(config_list):

    # create dir
    experiment_dir = os.path.join(os.getcwd(), str(config_i))
    os.makedirs(experiment_dir,exist_ok=True)

    os.chdir(experiment_dir)

    # dump config file
    with open("config.json", 'w') as outfile:
        commentjson.dump(config,outfile)
    
    # create job file
    if ENVIRONMENT == "slurm":
        job_file = generate_job_file(TRAINING_SCRIPT_FILE)
        with open('dota.job', 'w') as f:
            for line in job_file:
                f.write("%s\n" % line)
        subprocess.run(["sbatch", "dota.job"])


    elif ENVIRONMENT == "sge":
        job_file = generate_job_file_sge(TRAINING_SCRIPT_FILE)
        with open('dota.job', 'w') as f:
            for line in job_file:
                f.write("%s\n" % line)
        subprocess.run(["qsub", "dota.job"])

    elif ENVIRONMENT == "sge_debug":
        subprocess.run(["python", TRAINING_SCRIPT_FILE])
    elif ENVIRONMENT == "slurm_debug":
        subprocess.run(["python", TRAINING_SCRIPT_FILE])
    else:
        subprocess.run(["python", TRAINING_SCRIPT_FILE])

    

    # change directory back
    os.chdir(main_experiment_dir)
