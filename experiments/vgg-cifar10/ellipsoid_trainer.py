import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
import glob

import tabulate

import os
import sys
sys.path.append("../../simplex/")
import utils
from simplex_helpers import volume_loss
import time
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex
from simplex_models import SimplexModule

import scipy as scipy
from scipy import stats
import numpy as np
from re import L
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
import sys
sys.path.append("..")
import utils
from simplex_helpers import complex_volume

class EllipsoidLossFunction(nn.Module):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        # Create zero vector
        self.zero_vector = torch.zeros(dimensions)
        # Sample a ellipsoid matrix -> In this case the matrix corresponds to region with low loss
        self.ellipsoid_matrix = torch.tensor(scipy.stats.wishart(df=dimensions+2, scale=0.05*np.diag(np.ones((dimensions,)))).rvs())
    
    def get_ellipsoid_loss(self, X):
        # X is the sampled point
        # For example in the case of 2D space, X should be of the form ((x,y))

        # Sanity to reshape X to pre-defined format
        # print(X.shape)
        X = X.reshape([X.shape[0], 1])

        # Add zero padding so that X matches dimensions
        if X.shape[0] < self.dimensions:
            X = torch.cat((X, self.zero_vector[X.shape[0]:]), 0)

        # Multiply with ellipsoid matrix to generate a loss value for the given point
        transpose = torch.transpose(X, 0, 1).float()
        matrix1 = torch.matmul(transpose, self.ellipsoid_matrix.float()).float()
        matrix2 = torch.matmul(matrix1, X)

        # value < 1 , return -value
        # value > 1 , return value
        if matrix2[0][0] < 1:
            return -matrix2[0][0]
        else:
            return matrix2[0][0]

# Creates an ellipsoid loss region
ellipsoid_loss_function = EllipsoidLossFunction(2)

class EllipsoidLayer(SimplexModule):
    def __init__(self, fix_points):
        super(EllipsoidLayer, self).__init__(fix_points, ('weight', 'bias'))
        self.dimensions = 2
        # Random point
        # TODO: Can make this a random point from uniform distribution
        self.loss_point = torch.tensor([1.0] * self.dimensions)
        # TODO: Register a single parameter instead of cartesian vertices
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(self.loss_point, requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter('bias_%d' % i, None)

    def forward(self, input, coeffs_t):
        # Ignore input
        global ellipsoid_loss_function
        print(coeffs_t)
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        value = ellipsoid_loss_function.get_ellipsoid_loss(weight_t)
        return value
    
    # Getter and Setters for loss point
    def get_loss_point(self):
        return self.loss_point
    
    def set_loss_point(self, loss_point):
        self.loss_point = loss_point

# Optimiser: MSE
# Create dummy dataset when target = 1
class EllipsoidModel(nn.Module):
    def __init__(self, n_output, fix_points=[False]):
        super(EllipsoidModel, self).__init__()
        self.ellipseLayer = EllipsoidLayer(fix_points)
    
    def forward(self, x, coeffs_t):
        value = self.ellipseLayer(x, coeffs_t)
        return value


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import collections
class EllipsoidDataset(Dataset):
    def __init__(self, n):
        self.dataset = collections.defaultdict(int)
        for i in range(n):
            self.dataset[i] = torch.tensor(1.0)
        
    def __getitem__(self, index):
        return (torch.tensor(float(index)), self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)

def main(args):
    savedir = "saved-outputs/model_" + str(args.base_idx) + "/"
    from pathlib import Path
    Path(savedir).mkdir(parents=True, exist_ok=True)
    
    reg_pars = []
    out_dim = 1
    for ii in range(0, args.n_verts+2):
        fix_pts = [True]*(ii + 1)
        start_vert = len(fix_pts)
        simplex_model = SimplexNet(out_dim, EllipsoidModel, n_vert=start_vert, fix_points=fix_pts)
        log_vol = (simplex_model.total_volume() + 1e-4).log()
        reg_pars.append(max(float(args.LMBD)/log_vol, 1e-8))
    
    ## load in pre-trained model ##
    fix_pts = [False]
    n_vert = len(fix_pts)
    simplex_model = SimplexNet(out_dim, EllipsoidModel, n_vert=n_vert, fix_points=fix_pts)

    ## add a new points and train ##
    for vv in range(0, args.n_verts+1):
        simplex_model.add_vert()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        columns = ['vert', 'ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]
        dataset = EllipsoidDataset(1000)

        trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            time_ep = time.time()
            train_res = utils.train_epoch_volume(trainloader, simplex_model, criterion, optimizer, reg_pars[vv], args.n_sample)
            time_ep = time.time() - time_ep
            lr = optimizer.param_groups[0]['lr']
            values = [vv, epoch + 1, lr, train_res['loss'], train_res['accuracy'], time_ep, simplex_model.total_volume().item()]

            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)

        checkpoint = simplex_model.state_dict()
        fname = "simplex_vertex" + str(vv) + ".pt"
        torch.save(checkpoint, savedir + fname)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--LMBD",
        type=float,
        default=1e-10,
        metavar="lambda",
        help="value for \lambda in regularization penalty",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="verts",
        help="number of vertices in simplex",
    )
    parser.add_argument(
        "--n_verts",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )
    parser.add_argument(
        "--n_trial",
        type=int,
        default=5,
        metavar="N",
        help="number of simplexes to train",
    )
    parser.add_argument(
        "--base_idx",
        type=int,
        default=0,
        metavar="N",
        help="index of base model to use",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        metavar="N",
        help="evaluate every n epochs",
    )
    args = parser.parse_args()

    main(args)