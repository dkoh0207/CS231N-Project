import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn


class UResNet(torch.nn.Module):
    '''
    UResNet Pytorch Implementation, by Laura Domine.
    https://github.com/Temigo/uresnet_pytorch/blob/master/uresnet/models/uresnet_sparse.py
    '''
    def __init__(self, dim=3, size=192, nFeatures=16, depth=5, nClasses=5):
        import sparseconvnet as scn
        super(UResNet, self).__init__()
        #self._flags = flags
        dimension = dim
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = nFeatures  # Unet number of features
        nPlanes = [i*m for i in range(1, depth+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 1
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, size, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           scn.UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = torch.nn.Linear(m, nClasses)

    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        # (CS231N) We modify the following part to tailor the module for our purposes. 
        #coords = point_cloud[:, 0:-1].float()
        #features = point_cloud[:, -1][:, None].float()
        x = self.sparseModel(point_cloud)
        x = self.linear(x)
        return x

class ClusteringMLP(nn.Module):
    '''
    Three-Layer fully-connected network used for fine-tuning in transfer learning.
    Note that we decided to discard this method during experimentation.
    Author: Dae Heun Koh
    '''
    def __init__(self, input_dim=16, nHidden1=32, nHidden2=16, nClasses=3):
        super(ClusteringMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, nHidden1)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(nHidden1, nHidden2)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(nHidden2, nClasses)
        nn.init.kaiming_normal_(self.fc3.weight)
        
        self.bn_1 = nn.BatchNorm1d(nHidden1)
        self.bn_2 = nn.BatchNorm1d(nHidden2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn_1(self.fc1(x)))
        x = F.leaky_relu(self.bn_2(self.fc2(x)))
        x = self.fc3(x)
        return x

def get_unet(fname, dimension=3, size=192, nFeatures=16, depth=5, nClasses=5):
    '''
    Helper function for loading pretrained UResNet. 
    Author: Dae Heun Koh
    '''
    model = UResNet(dim=dimension, size=size, nFeatures=nFeatures, depth=depth, nClasses=nClasses)
    model = nn.DataParallel(model)
    checkpoint = torch.load(fname, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.module.sparseModel