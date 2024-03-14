
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
from model import make_rig_2_mesh_model

class ToyDataset():
    def __init__(self, rig_controls, vertices):
       
        self.rig_controls = rig_controls
        self.vertices = vertices   

        assert len(rig_controls) == len(vertices)
      
        self.length = len(rig_controls) 
        self.num_vertices = len(vertices[0])
        self.num_ctrl = len(rig_controls[0])
        
    def __getitem__(self, index):

        return torch.tensor(self.rig_controls[index]).float(), torch.tensor(self.vertices[index]).flatten().float()

    def __len__(self):

        return self.length    

def train():
    #
    # Train a rig approximation model using the dataset created in generate_toy_dataset.py
    # This model will be used to inverse the rig for a test animation in inverse_rig.py
    #

    current_folder = pathlib.Path(__file__).parent.resolve()
    checkpoint_folder = os.path.join(current_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, 'rig2mesh.pth.tar')

    model_shape = [1024, 1024]
    
    dataset_folder = os.path.join(current_folder, 'dataset')
    data = np.load(os.path.join(dataset_folder, 'dataset.npz'))
    #We keep 10% of the dataset as a validation set
    dataset = ToyDataset(data['arr_0'][:-100], data['arr_1'][:-100])
    validate_dataset = ToyDataset(data['arr_0'][100:], data['arr_1'][100:])

    print(len(dataset), 'training samples', len(validate_dataset), 'validation samples')

    #some pytorch version have a really slow dataloader for in-memory dataset so we arent using any workers
    dataloader = DataLoader(dataset, 32, num_workers=0, pin_memory=True, drop_last=False, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, 32, num_workers=0, pin_memory=True, drop_last=False, shuffle=False)

    model = make_rig_2_mesh_model(dataset.num_ctrl, model_shape, dataset.num_vertices*3)
    model.train()

    print('Rig Approximation model')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, eps=1e-9)

    epoch = 0
    best_val = 1000000
    early_stop_count = 0
    early_stop_patience = 50   
   
    def save_checkpoint():
        torch.save({           
            'model_state_dict': model.state_dict(),
            'model_shape': model_shape,
            'num_ctrl': dataset.num_ctrl,
            'num_vertices': dataset.num_vertices
        }, checkpoint_path)
 
    while True:
        num_batch = 0
        sum_loss = 0     
        sum_loss_validate = 0
        num_batch_validate = 0   
        model.train()

        for batch in tqdm(dataloader):

                ctrl_batch, vertex_batch = batch
            
                optimizer.zero_grad()
                out_vertex = model(ctrl_batch)

                loss = criterion(out_vertex, vertex_batch).mean()
                sum_loss += loss.item()
                num_batch += len(ctrl_batch)

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(validate_dataloader):
                
                ctrl_batch, vertex_batch = batch

                out_vertex = model(ctrl_batch)
                loss = criterion(out_vertex, vertex_batch).mean()
            
                sum_loss_validate += loss.item()
                num_batch_validate += len(ctrl_batch) 
                
        sum_loss /= num_batch
        sum_loss_validate /= num_batch_validate

        if (sum_loss_validate) < best_val:
            early_stop_count = 0
            best_val = (sum_loss_validate)
            save_checkpoint()
        elif early_stop_count >= early_stop_patience:
            break
        else:
            early_stop_count += 1

        print(epoch, "training loss", sum_loss, "validate loss", sum_loss_validate, optimizer.param_groups[0]['lr'], 'early_stop_count', early_stop_count)
        lr_scheduler.step(sum_loss)    

        epoch += 1

def test():

    # Always test using animations
    current_folder = pathlib.Path(__file__).parent.resolve()
    checkpoint_path = os.path.join(current_folder, 'checkpoints', 'rig2mesh.pth.tar')
    dataset_folder = os.path.join(current_folder, 'dataset')

    #re-load saved decoder model
    rig2mesh_checkpoint = torch.load(checkpoint_path)
    model_shape = rig2mesh_checkpoint['model_shape']
    num_ctrl = rig2mesh_checkpoint['num_ctrl']
    num_vertices = rig2mesh_checkpoint['num_vertices']
    model = make_rig_2_mesh_model(num_ctrl, model_shape, num_vertices*3)
    model.load_state_dict(rig2mesh_checkpoint['model_state_dict'])
    model.eval()

    #load test animation and ground truth
    anim = np.load(os.path.join(dataset_folder, 'anim.npy'))
    anim_4D = np.load(os.path.join(dataset_folder, 'anim_4D.npy'))

    #run animation through decoder model that learned to approximate the rig
    with torch.no_grad():
        found_4d = np.asarray([model(torch.tensor(frame).float()).detach().numpy().reshape(-1, 3) for frame in anim])

    average_vertex_error = np.linalg.norm(found_4d-anim_4D, axis=-1).mean()

    print('Euclidian rig approximation error:', average_vertex_error)
    print('This error bounds the rig inversion error.')
    print('So its important to get the rig approximation as accurate as possible')
    print('Fortunately, you should be able to use the rig to create as much training data as desired')

if __name__ == "__main__":

    train()
    test()