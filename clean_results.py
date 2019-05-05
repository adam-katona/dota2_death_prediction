

import sys
import os
import glob
import commentjson

RESULTS_PATH = [
#"/users/ak1774/scratch/esport/cluter_results/2019-03-12_18-30-12",
#  "/users/ak1774/scratch/esport/cluter_results/2019-03-12_16-40-00",
#                "/users/ak1774/scratch/esport/cluster_results"
    "/users/ak1774/scratch/esport/cluster_results/FixedAllChekpoint",
    #"/users/ak1774/scratch/esport/cluster_results/FixedMinimal"
#    "/users/ak1774/scratch/esport/cluster_results",
#    "/users/ak1774/scratch/esport/cluster_results",
#    "/users/ak1774/scratch/esport/cluster_results"

    ]
RESULT_NAME = "results_10_newwww.pickle"

import subprocess
def tail(file, n):
    with open(file) as f:
        content = f.readlines()
    return content[-n:]

log_files = []
for res_path in RESULTS_PATH:
    current_log_files = glob.glob(res_path + "/**/*.log", recursive=True)
    log_files.extend(current_log_files)
print(len(log_files))
dirs_with_results = [ os.path.dirname(os.path.abspath(log)) for log in log_files]

#print(log_files[108])

# look in the log file, for the last "Epoch done"
# look at the loss and the accuracy

data_points = []

for i,res_dir in enumerate(dirs_with_results):
    
    print(log_files[i])
    last_log_lines = tail(log_files[i],150)
    accuracy_found = False

    accuracy = None
    loss = None
    num_epoch = None

    for line in last_log_lines:
        #example: Epoch done  7623  loss:  2.397298  accuracy:  0.09747869318181818
        if "Epoch done" in line:
            split_line = line.split()
            accuracy = float(split_line[6])
            loss = float(split_line[4])
            num_epoch = int(split_line[2])
            accuracy_found = True
            
            break

    if accuracy_found == True:
        with open(res_dir + "/config.json") as f:
            config = commentjson.load(f)
            data_points.append((accuracy,loss,num_epoch,config,log_files[i]))
    else:
        print("No accuracy found: ",res_dir)

import pickle

with open(RESULT_NAME, 'wb') as f:
    pickle.dump(data_points, f)


    




