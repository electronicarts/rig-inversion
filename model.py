#
# Copyright (c) 2021-2024 Electronic Arts Inc. All Rights Reserved 
#

import torch.nn as nn
import numpy as np

def make_rig_2_mesh_model(num_input, shape, num_output):

    nn_layers = [nn.Linear(num_input, shape[0]), nn.LeakyReLU()]
    for layer0, layer1 in zip(shape[:-1], shape[1:]):
        nn_layers.append(nn.Linear(layer0, layer1))
        nn_layers.append(nn.LeakyReLU())

    nn_layers.append(nn.Linear(shape[-1], num_output))

    #Mesh values are unbounded so the last layer does not have an activation

    return nn.Sequential(*nn_layers)

def make_mesh_2_rig_model(num_parameters, shape, num_vertices_values):

    nn_layers = [nn.Linear(num_vertices_values, shape[0]), nn.LeakyReLU()]
    for layer0, layer1 in zip(shape[:-1], shape[1:]):
        nn_layers.append(nn.Linear(layer0, layer1))
        nn_layers.append(nn.LeakyReLU())

    nn_layers.append(nn.Linear(shape[-1], num_parameters))

    #Rig values are between 0 and 1 for the toy rig
    nn_layers.append(nn.Sigmoid())

    return nn.Sequential(*nn_layers)