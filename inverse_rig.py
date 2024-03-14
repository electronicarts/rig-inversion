#
# Copyright (c) 2021-2024 Electronic Arts Inc. All Rights Reserved 
#

import os
from tqdm import tqdm
import numpy as np
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import make_rig_2_mesh_model, make_mesh_2_rig_model
from rig import unit_cube, rig_model

def train():

    #
    # Train an encoder to inverse the rig in a self-supervised fashion
    # for a particular set of data 
    # using a pretrained differentiable rig approximation
    #

    current_folder = pathlib.Path(__file__).parent.resolve()
    dataset_folder = os.path.join(current_folder, 'dataset')
    checkpoint_folder = os.path.join(current_folder, 'checkpoints')
    rig2mesh_checkpoint_path = os.path.join(checkpoint_folder, 'rig2mesh.pth.tar')

    model_shape = [512]

    rig2mesh_checkpoint = torch.load(rig2mesh_checkpoint_path)       
    rig2mesh_model_shape = rig2mesh_checkpoint['model_shape']
    num_ctrl = rig2mesh_checkpoint['num_ctrl']
    num_vertices = rig2mesh_checkpoint['num_vertices']
   
    decoder_model = make_rig_2_mesh_model(num_ctrl, rig2mesh_model_shape, num_vertices*3)
    decoder_model.load_state_dict(rig2mesh_checkpoint['model_state_dict'])    
    decoder_model.eval()
    print('Decoder/rig approximation model')
    print(decoder_model)

    encoder_model = make_mesh_2_rig_model(num_ctrl, model_shape, num_vertices*3)
    encoder_model.train()
    print('Encoder/rig inversion model')
    print(encoder_model)

    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, eps=1e-6)

    ## This is the "capture" data we want to find rig parameters for
    anim_4D = np.load(os.path.join(dataset_folder, 'anim_4D.npy'))
    anim_4D_flatten = np.asarray([x.flatten() for x in anim_4D])

    dataloader = DataLoader(anim_4D_flatten, 32, num_workers=0, pin_memory=True, drop_last=False, shuffle=True)

    epoch = 0
    best_loss = 1000000
    early_stop_count = 0
    early_stop_patience = 50
 
    while True:
        num_batch = 0
        sum_loss = 0     

        for batch in tqdm(dataloader):

            # Train the encoder in a self-supervised fashion
            vertices = batch.float()
            optimizer.zero_grad()
            rig_ctrl = encoder_model(vertices)
            rig_output = decoder_model(rig_ctrl)
            loss = criterion(rig_output, vertices).mean() # loss is mesh to mesh, there is no loss on the rig parameters here.
            sum_loss += loss.item()
            num_batch += len(vertices)
            loss.backward()
            optimizer.step()        

        #This is an improvement not published in the paper. For most rigs, zero rig parameters are expected to produce a known neutral pose, and vice versa
        #We can use this to regularize the training by feeding the "neutral" mesh to the decoder and expect zero rig parameters.
        if epoch > 0:
            optimizer.zero_grad()
            zero_output = encoder_model(torch.tensor(np.expand_dims(unit_cube.flatten(),0)).float())
            loss = zero_output.mean()
            loss.backward()
            optimizer.step()    
                
        sum_loss /= num_batch

        if (sum_loss) < best_loss:
            early_stop_count = 0
            best_loss = (sum_loss)
        elif early_stop_count >= early_stop_patience:
            break
        else:
            early_stop_count += 1

        print(epoch, "training loss", sum_loss, 'lr', optimizer.param_groups[0]['lr'], 'early_stop_count', early_stop_count)
        lr_scheduler.step(sum_loss)    

        epoch += 1

    return encoder_model

def test(encoder_model):

    #
    # Test how successfull the rig inversion was by running the rig parameters produced by the encoder through the *actual* rig.
    #

    encoder_model.eval()

    current_folder = pathlib.Path(__file__).parent.resolve()
    dataset_folder = os.path.join(current_folder, 'dataset')
    anim_4D = np.load(os.path.join(dataset_folder, 'anim_4D.npy'))

    # Use the decoder to get rig parameters
    found_anim = np.array([encoder_model(torch.tensor(mesh.flatten()).float()).detach().numpy() for mesh in anim_4D])
    
    # to test we apply these rig parameters to the REAL rig
    found_4d = np.array([rig_model(frame) for frame in found_anim])

    average_vertex_error = dist = np.linalg.norm(found_4d-anim_4D, axis=-1).mean()

    print('Euclidian rig inversion error:', average_vertex_error)
    print('Unlike the decoder model which approximates the rig, the encoder model that inverse the rig is thrown away. A new model should be trained to inverse the rig on new data')

if __name__ == "__main__":

    encoder_model = train()
    test(encoder_model)