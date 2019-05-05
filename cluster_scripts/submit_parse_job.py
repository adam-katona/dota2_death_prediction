import sys
import os
import signal
import datetime
import subprocess

import shutil


def generate_job_file():

    lines = []
    lines.append("#!/bin/bash")
    lines.append("#SBATCH --job-name=dota_parse")
    lines.append("#SBATCH --ntasks=1")
    lines.append("#SBATCH --mem=5gb") # java virtual memory is fucked up
    lines.append("#SBATCH --time=05:00:00")
    lines.append("#SBATCH --output=parse_job_%j.log")
    lines.append("#SBATCH --account=CS-DCLABS-2019")
    lines.append("#SBATCH --export=ALL")
    lines.append("#SBATCH --array=0-499")    # Array range
    #lines.append("#SBATCH --partition=himem ")
    lines.append("")
    lines.append("")
    lines.append("echo `pwd`")
    lines.append("module load lang/Java/1.8.0_181")
    lines.append("export MALLOC_ARENA_MAX=4")
    lines.append("")
    lines.append("OMP_NUM_THREADS=1 python ./../parse_job.py")

    return lines






SCRIPT_ROOT_DIR = os.getcwd() 

# create and enter results dir
RESULTS_DIR = os.path.join(SCRIPT_ROOT_DIR,"parse_job_out")
os.makedirs(RESULTS_DIR,exist_ok=True)
os.chdir(RESULTS_DIR)

# create job file
job_file = generate_job_file()

print("Submtting job: ")
with open('dota_parse.job', 'w') as f:
    for line in job_file:
        print(line)
        f.write("%s\n" % line)

OUTPUT_DIR = os.path.join(RESULTS_DIR,"parsed_files")
os.makedirs(OUTPUT_DIR,exist_ok=True)

#submit job file
subprocess.run(["sbatch", "dota_parse.job"])


# get a list of filenames
# devide it into num worker groups
# write it to a file
# create job file
# crate folders, save git commit
# submit job

#parse.py

# go to directory
# read in replay list
# for each reaply file:
#   call java -jar entityrun.jar file_path
#   prerocess(gameName)
#   cleanup (delete csv)


#merge_and_normalize_data.py

# read all h5 files   (gonna run in high mem node if necessary)
# join them to one table
# normalize the data
# save it to one huge h5 file, or several large one, if it is too big to be in memory at once


#train.py

# read in config
# read in data
# init model
# while not stop:
#   x,y = getBatch()
#   model.train()
#   if time:
#       calculate test score
#       visualize test and train error, with graphs and histograms
















