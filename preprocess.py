

import os
import numpy as np
import pandas as pd
import math
import json

# NOTE
# It is important that everything is the same datatype in pandas, because otherwise getting values as an np array is expensive
# Use float32 everywhere 

def labels_to_indicies(labels):
    return [ i for i,label in labels]

def select_features_of_hero(hero_id,labels):
    hero_id_string = "player_" + str(hero_id) + "_"
    return [ (i,label) for i,label in labels if hero_id_string in label]
    
def select_features_by_name(name,labels):
    return [ (i,label) for i,label in labels if name in label]

def remove_features_by_name(name,data):
    return [ (i,label) for i,label in labels if name not in label]



def remove_paused_datapoints(data):

    time = np.diff(data["time"].values)
    is_paused = time < 0.0001 # time does not change, game is paused 
    data = data.drop(data.index[np.where(is_paused)])
    return data

# was this hero visible x second ago?
def add_historical_visibility_features(data):
    times = data.values[:,0]
    num_datapoints = data.shape[0]
    
    # estimate timestep
    delta_times = []
    for rand_i in np.random.randint(1,len(times),300):
        delta_times.append(times[rand_i] - times[rand_i-1])
    timestep = np.array(delta_times).mean()
    ticks_per_sec = int(math.ceil(1 / timestep))

    for hero_i in range(10):
        feature_name = "player_" + str(hero_i) + "_m_iTaggedAsVisibleByTeam"
        visibilities = data[feature_name].values

        for history_i in range(10):
            # visibility history_i+1 sec ago is a shifted version of visibility, with padeded zeros in the beginning
            new_feature = np.zeros(num_datapoints,dtype=np.float32)
            tick_diff = (history_i+1) * ticks_per_sec
            new_feature[tick_diff:] = visibilities[:-tick_diff]
            data["player_" + str(hero_i) + "_visibility_history_" + str(history_i)] = new_feature

    return data

# can be 0: alive, 1: dying, 2: dead
def life_state_to_times_of_death(life_data,is_one_dead):

    times = life_data.values[:,0]
    times_of_death_list = []
    for i in range(10):
        current_player_lifestate = life_data.values[:,i+1]

        if is_one_dead:
            current_player_lifestate[current_player_lifestate == 1] = 2
        else:
            current_player_lifestate[current_player_lifestate == 1] = 0

        diff = np.diff(current_player_lifestate)
        diff = np.insert(diff,0,0) # make diff the same length by inserting a 0 in front
        times_of_death = times[diff>0]
        times_of_death_list.append(times_of_death)


    return times_of_death_list
    

def create_time_until_next_death_stats(data,times_of_death_list):
    time_points = data.values[:,0]
    for i in range(10):

        death_times = times_of_death_list[i]
        time_to_next_death = np.full(len(time_points),-1,dtype=np.float32) # init to invalid
        next_death_time_label = "stat_" + str(i) + "_time_until_next_death"

        for time_i,time in enumerate(time_points):
            for death_time in death_times:
                if death_time - time > 0:
                    if time_to_next_death[time_i] < 0: # still invalid
                        time_to_next_death[time_i] = death_time - time
                    else:
                        time_to_next_death[time_i] = min(time_to_next_death[time_i], death_time - time)
                    
        data[next_death_time_label] = time_to_next_death

    return data

def create_who_dies_next_labels(data):

    labels = [(i,label) for i,label in enumerate(list(data))]
    next_death_indicies = labels_to_indicies(select_features_by_name("_time_until_next_death",labels))

    next_death_times = data.values[:,next_death_indicies]
    invalid_mask = next_death_times < 0  # invalid means the player will newer die in the rest of the time
    next_death_times += invalid_mask.astype(np.float32) * 1000000  # add a large number to invalid values 
    next_death_times = np.concatenate((next_death_times,np.full((data.shape[0],1),5)),axis=1) # add extra column. if this willbe the min, that means noone will die

    die_next_index = np.argmin(next_death_times,axis=1)

    for i in range(11):
        label_name = "label_who_dies_next_" + str(i)
        current_is_winner = die_next_index == i
        data[label_name] = current_is_winner.astype(np.float32)

    return data



def create_die_within_x_sec_feature(data,times_of_death_list,x):
    
    time_points = data.values[:,0]
    # new_features = np.zeros((len(time_points),10))
    
    for i in range(10):
        death_times = times_of_death_list[i]
        new_features = np.zeros(len(time_points),dtype=np.float32)  
        
        label_string = "label_" + str(i) + "_die_in_" + str(x)
        
        for time_i,time in enumerate(time_points):
            for death_time in death_times:
                if death_time > time and death_time - time < x:
                    new_features[time_i] = 1
                

        data[label_string] = new_features.astype(np.float32)
                    
    return data
        

def addPositionFeaturesTower(data):

    labels = [(i,label) for i,label in enumerate(list(data))]
    tower_labels = select_features_by_name("Tower_",labels)
    unique_tower_labels = select_features_by_name("m_lifeState",tower_labels) # this will return one label per tower
    unique_tower_labels = [label.replace("m_lifeState","") for i,label in unique_tower_labels]

    modified_data = data

    for tower_name in unique_tower_labels:
        
        cell_x = modified_data[tower_name + "CBodyComponent.m_cellX"].values
        cell_y = modified_data[tower_name + "CBodyComponent.m_cellY"].values
        vec_x = modified_data[tower_name + "CBodyComponent.m_vecX"].values
        vec_y = modified_data[tower_name + "CBodyComponent.m_vecY"].values
        
        pos_x = cell_x * 256 + vec_x
        pos_y = cell_y * 256 + vec_y
        
        modified_data[tower_name + "pos_x"] = pos_x.astype(np.float32)
        modified_data[tower_name + "pos_y"] = pos_y.astype(np.float32)
        
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_cellX",axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_cellY",axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_vecX",axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_vecY",axis=1)
    
    return modified_data


def addPositionFeatures(data):
    
    modified_data = data

    for hero_i in range(10):
        player_prefix = "player_" + str(hero_i) + "_"
        
        cell_x = modified_data[player_prefix + "CBodyComponent.m_cellX"].values
        cell_y = modified_data[player_prefix + "CBodyComponent.m_cellY"].values
        vec_x = modified_data[player_prefix + "CBodyComponent.m_vecX"].values
        vec_y = modified_data[player_prefix + "CBodyComponent.m_vecY"].values
        
        pos_x = cell_x * 256 + vec_x  # vec_x overflows at 256
        pos_y = cell_y * 256 + vec_y
        
        modified_data[player_prefix + "pos_x"] = pos_x.astype(np.float32)
        modified_data[player_prefix + "pos_y"] = pos_y.astype(np.float32)
        
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_cellX",axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_cellY",axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_vecX",axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_vecY",axis=1)

    return modified_data


def addHeroProximities(data):
    
    labels = [(i,label) for i,label in enumerate(list(data))]
    labels = select_features_by_name("pos",labels)
    labels = select_features_by_name("player_",labels)
    pos_x_indicies = labels_to_indicies(select_features_by_name("pos_x",labels))
    pos_y_indicies = labels_to_indicies(select_features_by_name("pos_y",labels))

    pos_x_vals = data.values[:,pos_x_indicies]
    pos_y_vals = data.values[:,pos_y_indicies]

    
    for hero_i in range(10):
        hero_team = int(hero_i / 5)
        current_ally_i = 0
        current_enemy_i = 0
        player_prefix = "player_" + str(hero_i) + "_"
        for other_hero_i in range(10):
            if other_hero_i == hero_i:
                continue

            feature_name = None
            other_hero_team = int(other_hero_i / 5)
            if hero_team == other_hero_team:
                feature_name = player_prefix + "ally_proximity_" + str(current_ally_i)
                current_ally_i += 1
            else:
                feature_name = player_prefix + "enemy_proximity_" + str(current_enemy_i)
                current_enemy_i += 1

            distances = np.sqrt((pos_x_vals[:,hero_i] - pos_x_vals[:,other_hero_i]) * (pos_x_vals[:,hero_i] - pos_x_vals[:,other_hero_i]) + 
                                (pos_y_vals[:,hero_i] - pos_y_vals[:,other_hero_i]) * (pos_y_vals[:,hero_i] - pos_y_vals[:,other_hero_i]))
            distances = np.minimum(distances,10000) # clamp the distances, it does not realy matter if it is so far away
            data[feature_name] = distances
    
    return data



def addClosestAliveTowers(data):

    labels = [(i,label) for i,label in enumerate(list(data))]
    team_2_tower_lables = select_features_by_name("Tower_2",labels)
    team_3_tower_lables = select_features_by_name("Tower_3",labels)

    team_2_tower_pos_x_labels = select_features_by_name("pos_x",team_2_tower_lables) 
    team_2_tower_pos_x_indicies = labels_to_indicies(team_2_tower_pos_x_labels)
    team_2_tower_pos_y_labels = select_features_by_name("pos_y",team_2_tower_lables) 
    team_2_tower_pos_y_indicies = labels_to_indicies(team_2_tower_pos_y_labels)
    team_2_tower_life_state_labels = select_features_by_name("m_lifeState",team_2_tower_lables) 
    team_2_tower_life_state_indicies = labels_to_indicies(team_2_tower_life_state_labels)

    team_3_tower_pos_x_labels = select_features_by_name("pos_x",team_3_tower_lables) 
    team_3_tower_pos_x_indicies = labels_to_indicies(team_3_tower_pos_x_labels)
    team_3_tower_pos_y_labels = select_features_by_name("pos_y",team_3_tower_lables)
    team_3_tower_pos_y_indicies = labels_to_indicies(team_3_tower_pos_y_labels)
    team_3_tower_life_state_labels = select_features_by_name("m_lifeState",team_3_tower_lables) 
    team_3_tower_life_state_indicies = labels_to_indicies(team_3_tower_life_state_labels)


    # NOTE
    # dont modify the data, because it will invalidate the indicies
    # modify it once everything is calculated

    closest_ally_tower = np.zeros((data.shape[0], 10),dtype=np.float32)
    closest_enemy_tower = np.zeros((data.shape[0], 10),dtype=np.float32)

    for team_iterator in range(2):
        team_index = team_iterator + 2  # first team is team 2, second team is team 3

        ally_tower_pos_x_indicies = team_2_tower_pos_x_indicies if team_index == 2 else team_3_tower_pos_x_indicies
        ally_tower_pos_y_indicies = team_2_tower_pos_y_indicies if team_index == 2 else team_3_tower_pos_y_indicies
        enemy_tower_pos_x_indicies = team_3_tower_pos_x_indicies if team_index == 2 else team_2_tower_pos_x_indicies
        enemy_tower_pos_y_indicies = team_3_tower_pos_y_indicies if team_index == 2 else team_2_tower_pos_y_indicies

        ally_tower_life_state_indicies = team_2_tower_life_state_indicies if team_index == 2 else team_3_tower_life_state_indicies
        enemy_tower_life_state_indicies = team_3_tower_life_state_indicies if team_index == 2 else team_2_tower_life_state_indicies

        ally_tower_pos_x = data.values[:,ally_tower_pos_x_indicies]
        ally_tower_pos_y = data.values[:,ally_tower_pos_y_indicies]

        enemy_tower_pos_x = data.values[:,enemy_tower_pos_x_indicies]
        enemy_tower_pos_y = data.values[:,enemy_tower_pos_y_indicies]

        ally_dead_mask = np.zeros((data.shape[0], 11),dtype=np.uint32)
        ally_dead_mask[:] = data.values[:,ally_tower_life_state_indicies] > 0.5

        enemy_dead_mask = np.zeros((data.shape[0], 11),dtype=np.uint32)
        enemy_dead_mask[:] = data.values[:,ally_tower_life_state_indicies] > 0.5

        for hero_iterator in range(5):
            hero_index = hero_iterator + 5 * team_iterator

            player_prefix = "player_" + str(hero_index) + "_"
            hero_pos_x = data[player_prefix + "pos_x"].values
            hero_pos_y = data[player_prefix + "pos_y"].values

            ally_tower_distances = np.sqrt((ally_tower_pos_x-hero_pos_x[:,np.newaxis]) * (ally_tower_pos_x-hero_pos_x[:,np.newaxis]) +
                                            (ally_tower_pos_y-hero_pos_y[:,np.newaxis]) * (ally_tower_pos_y-hero_pos_y[:,np.newaxis]))
            enemy_tower_distances = np.sqrt((enemy_tower_pos_x-hero_pos_x[:,np.newaxis]) * (enemy_tower_pos_x-hero_pos_x[:,np.newaxis]) +
                                            (enemy_tower_pos_y-hero_pos_y[:,np.newaxis]) * (enemy_tower_pos_y-hero_pos_y[:,np.newaxis]))

            # give a large value to dead towers, so they dont inflience the minimum
            ally_tower_distances = ally_tower_distances + ally_dead_mask * 10000000  
            enemy_tower_distances = enemy_tower_distances + enemy_dead_mask * 10000000  

            # 6000 is around quater the map length
            closest_ally_tower[:,hero_index] = np.minimum(ally_tower_distances.min(axis=1), 6000)
            closest_enemy_tower[:,hero_index] = np.minimum(enemy_tower_distances.min(axis=1), 6000)

    modified_data = data

    for hero_i in range(10):
        feature_name_prefix = "player_" + str(hero_i) + "_closest_tower_"
        modified_data[feature_name_prefix + "distance_ally"] = closest_ally_tower[:,hero_i]
        modified_data[feature_name_prefix + "distance_enemy"] = closest_enemy_tower[:,hero_i]
        
    # Delete all tower data
    all_tower_lables = select_features_by_name("Tower_",labels)
    for i,label in all_tower_lables:
        modified_data = modified_data.drop(label,axis=1)

    return modified_data



def addHeroOneHotEncoding(data):

    modified_data = data
    NUM_HEROS = 130 # Note this should be the max number of heroes

    for hero_i in range(10):
        
        hero_id_feature_name = "player_" + str(hero_i) + "_m_vecPlayerTeamData.000" + str(hero_i) + ".m_nSelectedHeroID"
        hero_id = data[hero_id_feature_name].values[0]
        hero_id_int = int(np.rint(hero_id))

        hero_one_hot = np.zeros(NUM_HEROS)
        hero_one_hot[hero_id_int] = 1

        for i in range(NUM_HEROS):
            feature_name = "player_" + str(hero_i) + "_hero_one_hot_" + str(i)
            modified_data[feature_name] = np.repeat(hero_one_hot[i], data.shape[0])

    return modified_data
    


def add_rate_of_change_features(data):

    # NOTE: rate of change features are depandant on the timestep, for now I don't care, because I will use the same timestep consistently
    # could devide by timestep in the future... 

    labels = [(i,label) for i,label in enumerate(list(data))]

    labels_to_make_diff = []
    diff_feature_name = []

    filtered_labels = select_features_by_name("pos_",labels)
    labels_to_make_diff.extend([label for i,label in filtered_labels])
    diff_feature_name.extend([label.replace("pos_","speed_") for i,label in filtered_labels])

    filtered_labels = select_features_by_name("_proximity_",labels)
    labels_to_make_diff.extend([label for i,label in filtered_labels])
    diff_feature_name.extend([label.replace("proximity","delta_proximity") for i,label in filtered_labels])

    filtered_labels = select_features_by_name("closest_tower_distance",labels)
    labels_to_make_diff.extend([label for i,label in filtered_labels])
    diff_feature_name.extend([label.replace("closest_tower_distance","delta_closest_tower_distance") for i,label in filtered_labels])

    filtered_labels = select_features_by_name("m_iHealth",labels)
    labels_to_make_diff.extend([label for i,label in filtered_labels])
    diff_feature_name.extend([label.replace("m_iHealth","delta_health") for i,label in filtered_labels])


    for label,new_label in zip(labels_to_make_diff,diff_feature_name):
        # take the diff and insert a zero in front 
        data[new_label] = np.insert(np.diff(data[label].values),0,0) 

    return data


from zlib import crc32

def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2**32

def str_to_float(s, encoding="utf-8"):
    return bytes_to_float(s.encode(encoding))

def add_game_name_hash(data,game_name):
    hash_val = str_to_float(game_name)
    data["stat_game_name_hash"] = np.repeat(hash_val, data.shape[0]).astype(np.float32)

    return data


def hero_id_to_roles(hero_id,hero_list,hero_roles_table):
    table_i = 0
    if hero_id < len(hero_list):
        hero_name = hero_list[hero_id][1]
        table_i = np.where(hero_roles_table["Hero"].values == hero_name)
    else:
        print("hero_role_failed",hero_id) 
    return hero_roles_table.values[table_i,1:].astype(np.float32).flatten()
    

def add_hero_role_features(data):
    
    JSON_PATH = '/users/ak1774/scratch/esport/death_prediction/heros.json'
    HERO_ROLE_CSV_PATH = "/users/ak1774/scratch/esport/death_prediction/Hero_Role_Data_Uptodate.csv"

    role_strings = ["Offlane","Mid","Support","Mage","RoamingSupport","SafelaneCarry"]
    
    with open(JSON_PATH) as f:
        heros_json = json.load(f)
    hero_list = [(item["id"],item["localized_name"]) for item in heros_json["heroes"]]
    hero_roles_table = pd.read_csv(HERO_ROLE_CSV_PATH)
    
    for hero_i in range(10):
        feature_name = "player_" + str(hero_i) + "_m_vecPlayerTeamData.000" + str(hero_i) + ".m_nSelectedHeroID"
        hero_id = data[feature_name].values[0]
        
        roles = hero_id_to_roles(int(hero_id),hero_list,hero_roles_table).flatten()
        if roles.size != len(role_strings):
            print("hero_role_failed 2: ",int(hero_id))
            roles = np.zeros(len(role_strings))
        roles = np.repeat(roles.reshape((1,-1)),data.shape[0],axis=0)
        for role_i,role_name in enumerate(role_strings):
            new_feature_name = "player_" + str(hero_i) + "_role_" + role_name
            data[new_feature_name] = roles[:,role_i]
    
    return data

def read_and_preprocess_data(game_name,sample=True):

    data_file_name = game_name + ".csv"
    life_stat_file_name = game_name + "_life.csv"

    data = pd.read_csv(data_file_name,dtype=np.float32)
    life_data = pd.read_csv(life_stat_file_name,dtype=np.float32)

    if data.isnull().values.any() == True:
        return None

    # for some reason we have 2 rows for each timestep in life_data
    # remove the duplicates
    life_data = life_data[~life_data["time"].duplicated(keep='first')] 

    data = remove_paused_datapoints(data)

    # get classification labels
    times_of_death_list = life_state_to_times_of_death(life_data,is_one_dead = True)
    data = create_time_until_next_death_stats(data,times_of_death_list)
    data = create_who_dies_next_labels(data)

    #data = create_die_within_x_sec_feature(data,times_of_death_list,2)
    data = create_die_within_x_sec_feature(data,times_of_death_list,5)
    data = create_die_within_x_sec_feature(data,times_of_death_list,7.5)
    data = create_die_within_x_sec_feature(data,times_of_death_list,10)
    data = create_die_within_x_sec_feature(data,times_of_death_list,12.5)
    data = create_die_within_x_sec_feature(data,times_of_death_list,15)
    data = create_die_within_x_sec_feature(data,times_of_death_list,17.5)
    data = create_die_within_x_sec_feature(data,times_of_death_list,20)
    #data = create_die_within_x_sec_feature(data,times_of_death_list,30)
    

    # body component to pos
    # also removes body component
    data = addPositionFeatures(data)
    data = addPositionFeaturesTower(data)

    # add closest tower features and remove all other tower features
    data = addClosestAliveTowers(data)

    data = addHeroProximities(data)

    data = add_rate_of_change_features(data)

    data = add_historical_visibility_features(data)

    data = add_game_name_hash(data,game_name)

    # data = add_hero_role_features(data)  # this was an experiment, not used in the final version


    # This is uses too much memory, this will be done for each batch during training
    # data = addHeroOneHotEncoding(data)

    print("Data shape: ", data.shape)
    if sample == True:
        return sample_data(data)
    return data



def sample_data(data,verbose = False):
    # sample the data
    # keep all the data when someone is about to die
    # keep a few times as much data when noone is about to die. 
    # Reather keep more than less, it is easy to downsample, but it is impossible to upsample
    # we dont want to keep evey datapoint without death, because the data would be very imbalanced (and want to save memory...)
    
    # calculate total number of death labels
    # take the same ammount of points when noone dies 
    # This way the data is still very imbalanced, since when someone dies, other people dont.
    # For calculating number of positive labels, use dies in 5 seconds labels
    anyone_about_to_die = np.array([False for _ in range(data.shape[0])])
    for i in range(10):
        label_name = "label_" + str(i) +"_die_in_20"
        anyone_about_to_die = np.logical_or(anyone_about_to_die, (data[label_name] > 0.5).values)

    # sample from points when noone is about to die
    num_datapoint_with_anyone_dying = sum(anyone_about_to_die)
    data_with_noone_dying = data[~anyone_about_to_die]
    data_with_noone_dying = data_with_noone_dying.sample(n=num_datapoint_with_anyone_dying, replace=False)

    # combine the dying and not dying
    sampled_data = pd.concat([data[anyone_about_to_die], data_with_noone_dying])

    if verbose:
        print("Original data shape: ", data.shape)
        print("Data points with anyone dying: ", data[anyone_about_to_die].shape)
        print("Datapoints taken with noone dying: ", data_with_noone_dying.shape)
        print("Ratio of data used: ", float(sampled_data.shape[0]) / data.shape[0])
        
    return sampled_data



import sys

if __name__ == "__main__":

    match_folder = ""
    output_folder = ""

    match_name = sys.argv[1]

    data = read_and_preprocess_data(match_name)
    data.to_hdf(match_name + '.h5', key='data', mode='w', complevel = 9,complib='zlib')

    
