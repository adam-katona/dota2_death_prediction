

import os
import numpy as np
import pandas as pd
import time
import sys
import math
import pickle
from sklearn.utils import shuffle

import preprocess



def run_cluster_calculate_norm_stats():

    H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
    H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_h5_files.txt"
    
    now = time.time()

    h5_files = get_h5_file_list(H5_PATH_PREFIX,H5_FILE_LIST)
    h5_files = np.random.choice(h5_files,150)
    data = load_data_chunk(h5_files,worker_id=None,num_workers=None)

    #data = load_all_data(H5_PATH_PREFIX,H5_FILE_LIST)
    print("Loading took: ", time.time()-now)
    print("Data shape: ", data.shape)
    sys.stdout.flush()

    now = time.time()
    norm_stats = calculate_normalization_stats(data)
    print("Collecting min max took: ", time.time()-now)
    sys.stdout.flush()

    now = time.time()
    with open("norm_stats.pickle", 'wb') as f:
        pickle.dump(norm_stats, f, pickle.HIGHEST_PROTOCOL)
    print("Pickleing took: ", time.time()-now)
    sys.stdout.flush()


def run_cluster_randomize(data_type):

    if data_type == "train":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/training_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/train/"

    elif data_type == "test":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/test_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/test/"

    elif data_type == "validation":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/validation_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/validation/"

    WORKER_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    NUM_WORKERS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

    now = time.time()
    h5_files = get_h5_file_list(H5_PATH_PREFIX,H5_FILE_LIST)
    data = load_data_chunk(h5_files,WORKER_ID,NUM_WORKERS)

    # shuffle the data
    data = shuffle(data)


    now = time.time()
    DATA_CHUNK_SIZE = 4000

    num_chunks = int(data.shape[0] / DATA_CHUNK_SIZE)
    rest = data.shape[0] - num_chunks * DATA_CHUNK_SIZE
    # the first rest chunk will contain 301 points, whis way every point is used, and they all have a similar size
    # NOTE I assume thet a worker have at least DATA_CHUNK_SIZE*DATA_CHUNK_SIZE datapoints, otherwise this tactic can fail...
    # Actually we just throw away a little bit of data, that is fine...
    current_index = 0
    for i in range(num_chunks):
        size = DATA_CHUNK_SIZE
        if i < rest:
            size += 1
        duta_chunk = data[current_index:current_index+size] 
        duta_chunk.to_hdf(OUT_FOLDER + 'data_chunk_' + str(WORKER_ID) + "_" + str(i) + '.h5', key='duta_chunk', mode='w', complevel = 9,complib='zlib')
        current_index += size

    print("Saving took: ", time.time()-now)
    sys.stdout.flush()


def run_cluster_normalize(data_type = None):

    if data_type == None:
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_h5_files.txt"
        OUT_FOLDER = "data_out/"

    elif data_type == "train":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/training_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/train/"

    elif data_type == "test":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/test_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/test/"

    elif data_type == "validation":
        H5_PATH_PREFIX = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/parse_job_out/parsed_files/"
        H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/cluster_scripts/validation_files.txt"
        OUT_FOLDER = "/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/validation/"
    
    WORKER_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    NUM_WORKERS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

    

    now = time.time()
    norm_stats = None
    with open("norm_stats.pickle", 'rb') as f:
        norm_stats = pickle.load(f)
    print("Unpickleing took: ", time.time()-now)
    sys.stdout.flush()

    now = time.time()
    h5_files = get_h5_file_list(H5_PATH_PREFIX,H5_FILE_LIST)
    data_chunk = load_data_chunk(h5_files,WORKER_ID,NUM_WORKERS)

    RECALCULATE_WHO_DIES_NEXT_LABELS = False 
    if RECALCULATE_WHO_DIES_NEXT_LABELS:
        data_chunk = preprocess.create_who_dies_next_labels(data_chunk)

    print("Loading took: ", time.time()-now)
    print("Data shape: ", data_chunk.shape)
    sys.stdout.flush()
    

    now = time.time()
    data = normalize_data(data_chunk,norm_stats)
    print("Normalizing took: ", time.time()-now)
    sys.stdout.flush()


    # shuffle the data
    data = shuffle(data)


    now = time.time()
    DATA_CHUNK_SIZE = 4000

    num_chunks = int(data.shape[0] / DATA_CHUNK_SIZE)
    rest = data.shape[0] - num_chunks * DATA_CHUNK_SIZE
    # the first rest chunk will contain 301 points, whis way every point is used, and they all have a similar size
    # NOTE I assume thet a worker have at least DATA_CHUNK_SIZE*DATA_CHUNK_SIZE datapoints, otherwise this tactic can fail...
    # Actually we just throw away a little bit of data, that is fine...
    current_index = 0
    for i in range(num_chunks):
        size = DATA_CHUNK_SIZE
        if i < rest:
            size += 1
        duta_chunk = data[current_index:current_index+size] 
        duta_chunk.to_hdf(OUT_FOLDER + 'data_chunk_' + str(WORKER_ID) + "_" + str(i) + '.h5', key='duta_chunk', mode='w', complevel = 9,complib='zlib')
        current_index += size

    print("Saving took: ", time.time()-now)
    sys.stdout.flush()


def get_h5_file_list(path_prefix,h5_file_list_path):

    with open(h5_file_list_path) as f:
        h5_files = f.readlines()

    h5_files = [x.strip() for x in h5_files]
    h5_files = [x.replace("./",path_prefix) for x in h5_files]
    return h5_files


def load_data_from_file(filename):
    return load_data_chunk([filename],worker_id=None,num_workers=None)


def load_all_data(path_prefix,h5_file_list_path):
    
    h5_files = get_h5_file_list(path_prefix,h5_file_list_path)
    return load_data_chunk(h5_files,worker_id=None,num_workers=None)


# will load all data if worker id is None
# TODO BUG: this is horribly inefficient. Concatenate will reallocate the whole dataset at every call...
# Solution: allocate a large memory beforehand... If dont know how big we need, use doubleing whenewer we run out? Maybe shrink in the end.
def load_data_chunk(h5_files,worker_id=None,num_workers=None):

    if worker_id is not None:
        files_per_worker = int(math.ceil(float(len(h5_files)) / num_workers))
        h5_files = h5_files[worker_id*files_per_worker : (worker_id+1)*files_per_worker]
        
    #h5_files = h5_files[0:10] # debug only take the first 10

    # read in all the data, and concatenate it into on big data frame
    data = None
    num_has_nan = 0

    for i,filename in enumerate(h5_files):
        if(i % 10) == 9:
            print("Loading file: ",i)
            sys.stdout.flush()
        #if data is not None:
        #    data.info()
        

        if data is None:
            data = pd.read_hdf(filename)
            if data.isnull().values.any():
                data = None
                num_has_nan += 1
                continue
        else:
            new_data = pd.read_hdf(filename)
            if new_data.isnull().values.any():
                num_has_nan += 1
                continue
            data = pd.concat((data,new_data),sort=False)

   # print("Ratio of corrupt files: ",float(num_has_nan) / len(h5_files))

    return data

def postprocess_data(data):
    data = preprocess.addHeroOneHotEncoding(data)
    return data

# These are called Correct because there was an incorrect version, and I did not want to break compatibility by reusing the same function name.
def getFeatureCorrectMinimal(data):

    only_include_list = []
    only_include_list.append("iHealth")
    only_include_list.append("iTotalEarnedGold")
    only_include_list.append("lifeState")

    only_include_list.append("_pos_")
    only_include_list.append("_proximity_")

    exclude_if_contains_list = []
    exclude_if_contains_list.append("ability") # to exclude ability level and stuff..
    exclude_if_contains_list.append("_delta_closest_tower_distance")
    exclude_if_contains_list.append("_delta_proximity_")

    return getFeatureIndicies(data,exclude_if_contains_list,only_include_list)



def getFeatureCorrectMedium(data):
    exclude_if_contains_list = []
    exclude_if_contains_list.append("_ability_") # all ability features
    exclude_if_contains_list.append("_hero_one_hot_") # hero id

    return getFeatureIndicies(data,exclude_if_contains_list)

def getFeatureCorrectAll(data):
    return getFeatureIndicies(data)


def getFeatureIndicies(data,exclude_if_contains_list = None,only_include_list = None):
    # get an example row
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)

    labels = [(i,label) for i,label in enumerate(list(example_row))]
    
    if only_include_list is not None:
        filtered_labels = []
        for i,label in labels:
            for include_label in only_include_list:
                if include_label in label:
                    filtered_labels.append((i,label))
        labels = filtered_labels


    if exclude_if_contains_list is not None:
        for exclude_pattern in exclude_if_contains_list:
            labels = [(i,label) for i,label in labels if exclude_pattern not in label]

    hero_feature_indicies = []
    for i in range(10):
        hero_labels = preprocess.select_features_of_hero(i,labels)
        hero_feature_indicies.append(preprocess.labels_to_indicies(hero_labels))
        hero_feature_indicies[-1].append(0) # dont forget the time

    return hero_feature_indicies

def getLableIndiciesWhoDiesNext(data):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name("who_dies_next",labels))
    return classification_label

def getLabelIndicies_die_in_n(data,label_name):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name(label_name,labels))
    return classification_label

def getLabelIndicies_die_in_5(data):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name("die_in_5",labels))
    return classification_label

def getLabelIndicies_die_in_10(data):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name("die_in_10",labels))
    return classification_label

def getLabelIndicies_die_in_15(data):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name("die_in_15",labels))
    return classification_label

def getLabelIndicies_die_in_20(data):
    example_row = data.sample(n=1,replace=False)
    example_row = postprocess_data(example_row)
    labels = [(i,label) for i,label in enumerate(list(example_row))]
    classification_label = preprocess.labels_to_indicies(preprocess.select_features_by_name("die_in_20",labels))
    return classification_label

# usage:
# small datasets:
#   data = normalize_data(data)
# large datasets:
#   stats = calculate_normalization_stats(data)
#   data_chunk = normalize_data(data_chunk,stats)
def normalize_data(data,normalization_stats = None):

    if normalization_stats is None:
        normalization_stats = calculate_normalization_stats(data)

    for feature_index,(label,min_value,max_value) in enumerate(normalization_stats):
        for hero_i in range(10):
            true_label = "player_" + str(hero_i) + label
            true_label = true_label.replace("TEAM_SLOT_IDX","000" + str(hero_i % 5))
            true_label = true_label.replace("PLAYER_IDX","000" + str(hero_i))
            
            if (max_value - min_value) == 0: # does not change, drop it
                data = data.drop(true_label,axis=1)
                if hero_i == 0:
                    print(true_label," is useless!!! It is dropped")
                
            else:
                # kwargs is weird, if I want to pass the value of the string reather than the name, i  must use the dictionary syntax...
                # reather than this: data = data.assign(true_label=(data[true_label] - min_value) / (max_value - min_value))
                data = data.assign(**{true_label : (data[true_label] - min_value) / (max_value - min_value)})
                
    return data


def calculate_normalization_stats(data):

    # NOTE there are also "label_" and stat_" labels (eg stat_0_time_until_next_death ) which we dont normalize
    # we dont use them as features

    # normalization takes too long, so we calculate min and max based on a fraction of the data
    representative_sample_size = min(10000,data.shape[0])
    take_every_n_th = int(math.floor(float(data.shape[0]) / representative_sample_size))

    # normalize time
    max_value = data["time"].max()
    min_value = data["time"].min()
    data = data.assign(time=(data["time"] - min_value) / (max_value - min_value))

    # normalize hero features
    # hero features are the same for every hero
    # we need to get the max and min looking at all the hero slots

    # dont normalize m_nSelectedHeroID, it is going to be turend into one hot encoding
    labels = [(i,label) for i,label in enumerate(list(data))]
    hero_labels = [label for i,label in preprocess.select_features_of_hero(0,labels) if "m_nSelectedHeroID" not in label ]
    hero_labels = [label.replace("player_0","") for label in hero_labels]
    hero_labels = [label.replace("0000","TEAM_SLOT_IDX") if ("m_vecDataTeam" in label) else label for label in hero_labels]
    hero_labels = [label.replace("0000","PLAYER_IDX") if ("m_vecPlayerTeamData" in label) else label for label in hero_labels]

    normalization_stats = []

    for label_i,label in enumerate(hero_labels):

        max_value = np.finfo(np.float32).min
        min_value = np.finfo(np.float32).max

        for hero_i in range(10):
            true_label = "player_" + str(hero_i) + label
            true_label = true_label.replace("TEAM_SLOT_IDX","000" + str(hero_i % 5))
            true_label = true_label.replace("PLAYER_IDX","000" + str(hero_i))

            max_value = max(max_value,data[true_label][::take_every_n_th].max())
            min_value = min(min_value,data[true_label][::take_every_n_th].min())
        # TODO BUG, this normalize excpects min max order, it does normalize kind of correctly: (val-max) / -range, (max becomes 0, min becomes 1)
        # Since it does not matter for neural networks, it was left unfixed.
        # when fixing it in the future: fic here, in normalize() and in test.py
        normalization_stats.append((label,max_value,min_value)) 
    return normalization_stats


# just get a random sample
# we dont care about getting balanced labels
def getBatchNaive(data,batch_size,hero_feature_indicies,classification_labels):

    data_batch = data.sample(n=batch_size,replace=False)

    # this is done only now, because it would takes up too much memory
    data_batch = postprocess_data(data_batch)

    num_features_per_hero = len(hero_feature_indicies[0])
    num_features_total = num_features_per_hero * 10
    #hero_features = np.zeros((batch_size,num_features_total))
    hero_features = []
    for i in range(10):
        #hero_features[:,(num_features_per_hero*i):(num_features_per_hero*(i+1))] = data_batch.values[:,hero_feature_indicies[i]]
        hero_features.append(data_batch.values[:,hero_feature_indicies[i]].astype(np.float32))

    classification_label_values = data_batch.values[:,classification_labels].astype(np.float32)

    return hero_features,classification_label_values

def getSequencialNaive(data,hero_feature_indicies,classification_labels):

    data_batch = data#data.sample(n=10000,replace=False)

    # this is done only now, because it would takes up too much memory
    data_batch = postprocess_data(data_batch)

    hero_features = []
    for i in range(10):
        #hero_features[:,(num_features_per_hero*i):(num_features_per_hero*(i+1))] = data_batch.values[:,hero_feature_indicies[i]]
        hero_features.append(data_batch.values[:,hero_feature_indicies[i]].astype(np.float32))

    classification_label_values = data_batch.values[:,classification_labels].astype(np.float32)

    return hero_features,classification_label_values



def getBalancedBatchForPlayer(data,player_i,batch_size,hero_feature_indicies,classification_labels,get_death_times=False):

    # get a batch, where half of the time the selected player dies, the other half not
    #player_dies_mask = data["label_who_dies_next_" + str(player_i)].values > 0.5
    # classification label indicies is contains the indicies of labels like "player_0_die_in_10"
    player_dies_mask = data.values[:,classification_labels[player_i]] > 0.5

    num_sample_from_die = int(batch_size/2)
    num_sample_from_not_die = batch_size - num_sample_from_die

    have_enough_unique_data = sum(player_dies_mask) > num_sample_from_die 
    data_batch_die = data[player_dies_mask].sample(n=num_sample_from_die,replace=(have_enough_unique_data == False))

    have_enough_unique_data = sum(~player_dies_mask) > num_sample_from_not_die 
    data_batch_not_die = data[~player_dies_mask].sample(n=num_sample_from_not_die,replace=(have_enough_unique_data == False))

    data_batch = pd.concat([data_batch_die,data_batch_not_die])

    # this is done only now, because it would takes up too much memory
    data_batch = postprocess_data(data_batch)

    hero_features = []
    for i in range(10):
        hero_features.append(data_batch.values[:,hero_feature_indicies[i]].astype(np.float32))

    classification_label_values = data_batch.values[:,classification_labels].astype(np.float32)

    if get_death_times == True:
        labels = [(i,label) for i,label in  enumerate(list(data))]
        death_time_indicies = preprocess.labels_to_indicies(preprocess.select_features_by_name("time_until_next_death",labels))
        death_times = data_batch.values[:,death_time_indicies]
        return hero_features,classification_label_values,death_times

    return hero_features,classification_label_values


def getBatchBalanced(data,batch_size,hero_feature_indicies,classification_labels,get_death_times=False):

    no_one_dies_mask = data["label_who_dies_next_10"].values > 0.5

    num_sample_from_die = int(batch_size * 10.0/11)
    num_sample_from_not_die = batch_size - num_sample_from_die

    have_enough_unique_data = sum(~no_one_dies_mask) > num_sample_from_die 
    data_batch_die = data[~no_one_dies_mask].sample(n=num_sample_from_die,replace=(have_enough_unique_data == False))

    have_enough_unique_data = sum(no_one_dies_mask) > num_sample_from_not_die 
    data_batch_not_die = data[no_one_dies_mask].sample(n=num_sample_from_not_die,replace=(have_enough_unique_data == False))


    data_batch = pd.concat([data_batch_die,data_batch_not_die])

    # this is done only now, because it would takes up too much memory
    data_batch = postprocess_data(data_batch)

    hero_features = []
    for i in range(10):
        hero_features.append(data_batch.values[:,hero_feature_indicies[i]].astype(np.float32))

    classification_label_values = data_batch.values[:,classification_labels].astype(np.float32)

    if get_death_times == True:
        labels = [(i,label) for i,label in  enumerate(list(data))]
        death_time_indicies = preprocess.labels_to_indicies(preprocess.select_features_by_name("time_until_next_death",labels))
        death_times = data_batch.values[:,death_time_indicies]
        return hero_features,classification_label_values,death_times

    return hero_features,classification_label_values
















