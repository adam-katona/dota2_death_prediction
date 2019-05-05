

import sys
import os
import math
import subprocess

import numpy as np
import pandas as pd
import math
import json
import pickle

sys.path.append("/users/ak1774/scratch/esport/death_prediction")
import preprocess
import data_loader


ENVIRONMENT = "viking"
#ENVIRONMENT = "local"

RESULTS_DIR = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/test_whole_matches"
os.makedirs(RESULTS_DIR,exist_ok=True)
os.chdir(RESULTS_DIR)


MATCH_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/testing/test_dems.txt"
#MATCH_FILE_LOCATION_PREFIX = "/mnt/lustre/groups/cs-dclabs-2019/esport/gamefiles/"
MATCH_FILE_LOCATION_PREFIX = "/users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/"
with open(MATCH_FILE_LIST) as f:
    all_match_files = f.readlines()



if True:
    WORKER_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    NUM_WORKERS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
else:
    WORKER_ID = 1

JAR_PATH = "/users/ak1774/scratch/esport/death_prediction/parser/entityrun.one-jar.jar"
NORM_STATS_FILE = "/users/ak1774/scratch/esport/death_prediction/norm_stats.pickle"


DEM_FILE = all_match_files[WORKER_ID]
DEM_FILE = DEM_FILE.strip()
            # /users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/pro/demP/4262403495_1586866351.dem/4262403495_1586866351.dem
#DEM_FILE = "/users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/pro/demP/3475430774_414202435.dem/3475430774_414202435.dem"
DEM_FILE = DEM_FILE.replace("./",MATCH_FILE_LOCATION_PREFIX)

match_name = DEM_FILE.split("/")[-1].replace(".dem","")

print(DEM_FILE)

# parse dem file
subprocess.run(["java","-jar",JAR_PATH,DEM_FILE,RESULTS_DIR])
print("JAVA finished")
sys.stdout.flush()


data = preprocess.read_and_preprocess_data(match_name,sample=False)
if data is None:
    print("Corrupt data, did the parser failed?")
else:
    print("Data preprocessed.")

CSV_PAPTH = os.path.join(RESULTS_DIR,match_name + ".csv")
CSV_LIFE_PAPTH = os.path.join(RESULTS_DIR,match_name + "_life.csv")
subprocess.run(["rm",CSV_PAPTH])
subprocess.run(["rm",CSV_LIFE_PAPTH])
print("CSV deleted")
sys.stdout.flush()

if data is not None:

    norm_stats = None
    with open(NORM_STATS_FILE, 'rb') as f:
        norm_stats = pickle.load(f)

    data = data_loader.normalize_data(data,norm_stats)
    print("Data normalized.")

    data.to_hdf(match_name + "_whole_match_normalized.h5", key='data', mode='w', complevel = 1,complib='zlib')
