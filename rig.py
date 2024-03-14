
#
# Copyright (c) 2021-2024 Electronic Arts Inc. All Rights Reserved 
#

from scipy.spatial.transform import Rotation as R
import numpy as np

unit_cube = np.asarray([ [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, -1, -1]])

def rig_model(rig_parameters):

    # return vertices values from rig parameters
    # This toy rig is a simple unit cube mesh deformed with random axis-scale and rotation

    sx = R.from_euler('x', (rig_parameters[0]-0.5)*90, degrees=True)
    sy = R.from_euler('y', (rig_parameters[1]-0.5)*90, degrees=True)
    sz = R.from_euler('z', (rig_parameters[2]-0.5)*90, degrees=True)

    scale = np.zeros((3,3))
    scale[0,0] =  (rig_parameters[3]*2.5)-0.5
    scale[1,1] =  (rig_parameters[4]*2.5)-0.5
    scale[2,2] =  (rig_parameters[5]*2.5)-0.5
    

    # Convention of order of operation is unimportant here since we are making a toy blackbox rig
    mesh = unit_cube
    mesh = np.matmul(mesh, scale)
    mesh = sx.apply(mesh)
    mesh = sy.apply(mesh)
    mesh = sz.apply(mesh)

    return mesh