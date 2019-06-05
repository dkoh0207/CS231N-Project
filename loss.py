import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn


class DiscriminativeLoss(torch.nn.Module):
    '''
    Implementation of the Discriminative Loss Function in Pytorch
    Note that there are many other implementation in Github, yet we decided to
    implement from scratch for practice and to tailor to our purposes. 
    '''
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, 
                 alpha=1.0, beta=1.0, gamma=0.001,
                 use_gpu=False, multiclass=False):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_gpu = use_gpu
        self.multiclass = multiclass
        
    def find_cluster_means(self, features, label):
        '''
        For a given image, compute the mean clustering point mu_c for each
        cluster label in the feature dimension.
        '''
        n_clusters = label.unique().size()
        cluster_labels = list(label.unique(sorted=True).numpy())
        # Ordering of the cluster means are crucial.
        cluster_means = []
        for c in cluster_labels:
            index = (label == c).squeeze(1).nonzero()
            index = index.squeeze(1)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        #print(cluster_means)
        return cluster_means
        
    def variance_loss(self, features, label, cluster_means, margin=1):
        var_loss = 0
        n_clusters = len(cluster_means)
        cluster_labels = list(label.unique(sorted=True).numpy())
        for i, c in enumerate(cluster_labels):
            index = (label == c).squeeze(1).nonzero()
            index = index.squeeze(1)
            dists = torch.norm(features[index] - cluster_means[i], p=self.norm, dim=1)
            hinge = torch.clamp(dists-1, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            var_loss += l
        var_loss /= n_clusters
        #print(var_loss)
        return var_loss
    
    def mean_distance_loss(self, cluster_means, margin=2):
        mean_loss = 0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            return 0
        for i, c1 in enumerate(cluster_means):
            for j, c2 in enumerate(cluster_means):
                if i != j:
                    dist = torch.norm(c1 - c2, p=self.norm)
                    hinge = torch.clamp(2.0 * margin - dist, min=0)
                    mean_loss += torch.pow(hinge, 2)
        if n_clusters > 1:
            mean_loss /= (n_clusters - 1) * n_clusters
        #print(mean_loss)
        return mean_loss
    
    def regularization(self, cluster_means, norm=2):
        reg = 0
        n_clusters, feature_dim = cluster_means.shape
        for i in range(n_clusters):
            #print(torch.norm(cluster_means[i, :], norm))
            reg += torch.norm(cluster_means[i, :], p=norm)
        #print(reg)
        reg /= n_clusters
        #print(reg)
        return reg
    
    def combine(self, features, label):
        
        c_means = self.find_cluster_means(features, label)
        loss_dist = self.mean_distance_loss(c_means, margin=self.delta_dist)
        loss_var = self.variance_loss(features, label, c_means, margin=self.delta_var)
        loss_reg = self.regularization(c_means, norm=self.norm)
        
        loss = self.alpha * loss_var + self.beta * loss_dist + self.gamma * loss_reg
        
        return loss
    
    def forward(self, x, y, seg_labels=None):
        
        if self.multiclass:
            loss = []
            assert seg_labels is not None
            semantic_classes = seg_labels.unique()
            for sc in semantic_classes:
                index = (seg_labels == sc).squeeze(1).nonzero()
                index = index.squeeze(1)
                x_c, y_c = x[index], y[index]
                loss.append(self.combine(x_c, y_c))
            return sum(loss)
        else:
            return self.combine(x,y)