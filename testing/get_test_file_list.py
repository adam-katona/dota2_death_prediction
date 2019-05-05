
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




ALL_DEM_FILES = "/users/ak1774/scratch/esport/death_prediction/all_dem_files.txt"
TEST_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/test_files.txt"

# read the test file list (which is processed alread, and find the corresponding )
with open(ALL_DEM_FILES) as f:
    all_match_files = f.readlines()

all_match_files = [x.strip() for x in all_match_files] 
test_match_files = []

with open(TEST_FILE_LIST) as f:
    parsed_test_files = f.readlines()

for i,line in enumerate(parsed_test_files):
    line = line.strip()
    line = line.replace(".h5","")
    line = line.replace("./","")

    print(i)

    for dem_line in all_match_files:
        if line in dem_line:
            test_match_files.append(dem_line)

with open('test_dems.txt', 'w') as f:
    for item in test_match_files:
        f.write("%s\n" % item)