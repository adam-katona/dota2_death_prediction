

import os
import numpy as np
import pandas as pd

import preprocess
import data_loader

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F




class LinearRegression(torch.nn.Module):
    def __init__(self,num_features,num_labels):
        super(LinearRegression, self).__init__()
        
        self.linear = torch.nn.Linear(in_features=num_features,out_features=num_labels)

    def forward(self, hero_features):
        x = torch.cat(hero_features,1)
        x = self.linear(x)
        return x

    
class SimpleFF(torch.nn.Module):
    def __init__(self,num_features,num_labels):
        super(SimpleFF, self).__init__()
        
        self.linear1 = torch.nn.Linear(in_features=num_features,out_features=200)
        self.linear2 = torch.nn.Linear(in_features=200,out_features=100)
        self.linear3 = torch.nn.Linear(in_features=100,out_features=num_labels)

    def forward(self, hero_features):
        x = torch.cat(hero_features,1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SharedHeroWeightsFF(torch.nn.Module):
    def __init__(self,num_features_per_hero,num_labels,shared_layer_sizes=None,final_layer_sizes=None):
        super(SharedHeroWeightsFF, self).__init__()
        
        if shared_layer_sizes is None:
            shared_layer_sizes = [100,60,20]
        if final_layer_sizes is None:
            final_layer_sizes = [100]

        self.shared_layers = nn.ModuleList([]) # have to use ModuleList because using a plain list fails to populate model.parameters()
        self.final_layers = nn.ModuleList([])

        previous_layer_size = num_features_per_hero
        for layer_size in shared_layer_sizes:
            self.shared_layers.append(torch.nn.Linear(in_features=previous_layer_size,out_features=layer_size))
            previous_layer_size = layer_size

        previous_layer_size = 10 * previous_layer_size # this is the size after the concatenation
        for layer_size in final_layer_sizes:
            self.final_layers.append(torch.nn.Linear(in_features=previous_layer_size,out_features=layer_size))
            previous_layer_size = layer_size
        
        # add the last layer
        self.final_layers.append(torch.nn.Linear(in_features=previous_layer_size,out_features=num_labels))

    def forward(self, hero_features):
        vals = []
        for hero_feature in hero_features:
            hero_x = hero_feature
            for shared_layer in self.shared_layers:
                hero_x = torch.relu(shared_layer(hero_x))
            vals.append(hero_x)
            
        x = torch.cat(vals,dim=1)
            
        for layer_i,final_layer in enumerate(self.final_layers):
            x = final_layer(x)
            if layer_i < (len(self.final_layers)-1): # no ReLU for the last layer
                x = torch.relu(x)

        return x