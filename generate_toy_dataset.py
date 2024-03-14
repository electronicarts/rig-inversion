#
# Copyright (c) 2021-2024 Electronic Arts Inc. All Rights Reserved 
#

import os
import numpy as np
import pathlib
from rig import unit_cube, rig_model

def generate_dataset(num_samples=1000):
    ###
    ### Generate a simple toy dataset of vertices by applying random transformations to a unit cube
    ### These transformations refine a toy rig of transform coordinates to vertices
    ### In a real use case this dataset would be generated from running the actual animation rig
    ###

    current_folder = pathlib.Path(__file__).parent.resolve()
    rig_parameters_dataset = []
    mesh_vertices_dataset = []

    for sample in range(num_samples):

        #all rig parameters of our toy rig are [0,1]
        rig_parameters = np.random.rand(6)

        mesh = rig_model(rig_parameters)
        
        rig_parameters_dataset.append(rig_parameters)
        mesh_vertices_dataset.append(mesh)
    
    dataset_folder = os.path.join(current_folder, 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    np.savez(os.path.join(dataset_folder, 'dataset.npz'), rig_parameters_dataset, mesh_vertices_dataset)    
    
    ###
    ### We also generate a smooth synthetic animation
    ### This would normally be captured or generated animation we wish to find right parameters for.    
    ###

    anim_length = 128

    start = np.random.rand(6)
    end =  np.random.rand(6)
    deltas = end-start
    anim = [start + (x/anim_length)*deltas for x in range(anim_length)]
    anim_4D = [rig_model(rig_parameters) for rig_parameters in anim]

    np.save(os.path.join(dataset_folder, 'anim.npy'), anim)
    np.save(os.path.join(dataset_folder, 'anim_4D.npy'), anim_4D)

if __name__ == "__main__":

    generate_dataset()