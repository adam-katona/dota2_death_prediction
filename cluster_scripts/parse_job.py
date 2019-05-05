

import sys
import os
import math
import subprocess
import pickle
import time

sys.path.append("/users/ak1774/scratch/esport/death_prediction")

import preprocess
import data_loader

# command to delete corrupt files
# grep -r --include="*.log" "PARSING_ERROR: " ./../ |  awk 'NF>1{print $NF}' | awk '$0=$0".h5"' | xargs rm



# Before run:
# generate dem file list with:
# find . -name "*.dem" -type f > /users/ak1774/scratch/esport/death_prediction/all_dem_files.txt
# then set MATCH_FILE_LOCATION_PREFIX

MATCH_FILE_LOCATION_PREFIX = "/users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/"
#MATCH_FILE_LOCATION_PREFIX = "/users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/semipro/"
#MATCH_FILE_LOCATION_PREFIX = "/users/ak1774/scratch/cs-dclabs-2019/esport/gamefiles/pro/"

#MATCH_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_pro_dems.txt"
#MATCH_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_semipro_dems.txt"
#MATCH_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/some_dem_files.txt"
MATCH_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_dem_files.txt"


JAR_PATH = "/users/ak1774/scratch/esport/death_prediction/parser/entityrun.one-jar.jar"


ALSO_NORMALIZE = True
NORM_STATS_FILE = "/users/ak1774/scratch/esport/death_prediction/norm_stats.pickle"


norm_stats = None
with open(NORM_STATS_FILE, 'rb') as f:
    norm_stats = pickle.load(f)


# debug
WORKER_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
NUM_WORKERS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
#WORKER_ID = 1
#NUM_WORKERS = 8

print("I am worker ",WORKER_ID," from ",NUM_WORKERS)


ROOT_DIR = os.getcwd() 
RESULTS_DIR = os.path.join(ROOT_DIR,"parsed_files")


with open(MATCH_FILE_LIST) as f:
    all_match_files = f.readlines()

all_match_files = [x.strip() for x in all_match_files] 

num_matches = len(all_match_files)
match_per_worker = int(math.ceil(float(num_matches) / NUM_WORKERS))
print("Match per worker: ",match_per_worker)
sys.stdout.flush()


first_match_index_for_this_task = match_per_worker * WORKER_ID
print("My first match is  ",first_match_index_for_this_task, " num matches is ",num_matches)
sys.stdout.flush()
for i in range(match_per_worker):
    match_index = first_match_index_for_this_task + i
    if match_index >= num_matches:
        break # done

    
    
    match_path = all_match_files[match_index]
    match_path = match_path.replace("./",MATCH_FILE_LOCATION_PREFIX)
    match_name = match_path.split("/")[-1].replace(".dem","")

    print("Parsing match number ",match_index," ",match_name)
    sys.stdout.flush()
    #print(match_path)

    subprocess.run(["java","-jar",JAR_PATH,match_path,RESULTS_DIR])
    print("JAVA finished")
    sys.stdout.flush()

    os.chdir(RESULTS_DIR)

    data = preprocess.read_and_preprocess_data(match_name)
    if data is not None:
        if ALSO_NORMALIZE == True:
            now = time.time()
            data = data_loader.normalize_data(data,norm_stats)
            print("Normalizing took: ", time.time()-now)
            sys.stdout.flush()

        data.to_hdf(match_name + '.h5', key='data', mode='w', complevel = 1,complib='zlib')
        print("H5 created")
    else:
        print("Corrupt match deleted: ",match_name)
    sys.stdout.flush()

    CSV_PAPTH = os.path.join(RESULTS_DIR,match_name + ".csv")
    CSV_LIFE_PAPTH = os.path.join(RESULTS_DIR,match_name + "_life.csv")
    subprocess.run(["rm",CSV_PAPTH])
    subprocess.run(["rm",CSV_LIFE_PAPTH])
    print("CSV deleted")
    sys.stdout.flush()


