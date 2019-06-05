import torch
import torch.nn as nn
import glob
import os.path as osp
import numpy as np

import sys
sys.path.append("../new_notebooks/ipynb/dlp_opendata_api")
sys.path.append("../new_notebooks/ipynb")
from osf.image_api import image_reader_3d
from osf.particle_api import *
from osf.cluster_api import *

from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


class ClusteringAEData(Dataset):
    """
    A customized data loader for clustering.
    """
    def __init__(self, root, numPixels=192, filenames=None):
        """
        Initialize Clustering Dataset

        Inputs:
            - root: root directory of dataset
            - preload: if preload dataset into memory.
        """
        self.cluster_filenames = []
        self.energy_filenames = []
        self.root = root
        self.numPixels = str(numPixels)
        
        if filenames:
            self.energy_filenames = filenames[0]
            self.cluster_filenames = filenames[1]
            print(self.energy_filenames)

        self.energy_filenames.sort()
        self.cluster_filenames.sort()
        self.cluster_reader = cluster_reader(*self.cluster_filenames)
        self.energy_reader = image_reader_3d(*self.energy_filenames)
        self.len = self.energy_reader.entry_count()
        assert self.len == self.cluster_reader.entry_count()

    def __getitem__(self, index):
        """
        Get a sample from dataset.
        """
        voxel, ins_label = self.cluster_reader.get_image(index)
        _, energy, seg_label = self.energy_reader.get_image(index)
        voxel, ins_label = torch.from_numpy(voxel), torch.from_numpy(ins_label)
        seg_label = torch.from_numpy(seg_label)
        seg_label = torch.unsqueeze(seg_label, dim=1).type(torch.LongTensor)
        energy = torch.from_numpy(energy)
        energy = torch.unsqueeze(energy, dim=1)
        ins_label = torch.unsqueeze(ins_label, dim=1).type(torch.LongTensor)
        voxel = voxel.cuda()
        energy = energy.cuda()
        #with torch.no_grad():
        #    out = unet((voxel, energy))
        return (voxel, energy), ins_label, seg_label

    def __len__(self):
        """
        Total number of sampels in dataset.
        """
        return self.len

def ae_collate(batch):
    """
    Custom collate_fn for the Clustering dataset.
    Author: Dae Heun Koh
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def compute_accuracy(embedding, truth, bandwidth=0.5):
    '''
    Compute Adjusted Rand index score (accuracy) for given embedding. 
    Inputs:
        embedding: torch array with coordinates
        truth: truth labels torch for cluster assignments
    Author: Mingyu Kang, Dae Heun Koh
    '''
    embed = embedding.cpu()
    embed = embed.detach().numpy()
    th = truth.numpy().squeeze()
    embed = np.atleast_1d(embed)
    th = np.atleast_1d(th)
    with torch.no_grad():
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit_predict(embed)
        score = adjusted_rand_score(clustering, th)
        return score, clustering
    
def compute_accuracy_with_segmentation(embedding, truth, seg_labels):
    '''
    Compute accuracy by masking on semantic segmentation labels.
    Author: Dae Heun Koh
    '''
    acc = []
    semantic_classes = seg_labels.unique()
    for sc in semantic_classes:
        index = (seg_labels == sc).squeeze(1).nonzero()
        index = index.squeeze(1)
        embedding_c, truth_c = embedding[index], truth[index]
        acc.append(compute_accuracy(embedding_c, truth_c))
    return sum(acc) / float(len(acc))

def save_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    '''
    Checkpoint saving helper function, from CS231N Pytorch tutorial.
    Minor modifications are added to include the learning rate scheduler.
    '''
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    '''
    Checkpoint loading helper function, from CS231N Pytorch tutorial. 
    Minor modifications are added to include the learning rate scheduler.
    '''
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded from %s' % checkpoint_path)

def test(model, devloader, criterion, batch_size):
    '''
    Helper function for testing model on validation set, given by devloader. 
    Author: Dae Heun Koh
    '''
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for k, batch in enumerate(devloader):
            x_batch = batch[0]
            y_batch = batch[1]
            for j, data in enumerate(x_batch):
                out = model(data)
                loss = criterion(out, y_batch[j])
                acc = compute_accuracy(out, y_batch[j])
                test_loss += loss
                test_acc += acc
    return test_loss.item() / float(batch_size), test_acc / float(batch_size)