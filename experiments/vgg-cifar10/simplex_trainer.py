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
    
    # Function returns True if it's within the ellipsoid else False
    def get_ellipsoid_loss(self, X):
        # X is the sampled point (chosen at random from the entire space)
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
        # print(matrix2[0][0])

        if matrix2[0][0] < 5:
            return matrix2[0][0], True
        else:
            return matrix2[0][0], False

# Creates a layer with all cartesian variables (vertices)
class EllipsoidLayer(SimplexModule):
    def __init__(self, fix_points):
        super(EllipsoidLayer, self).__init__(fix_points, ('cartesian_vertices'))
        self.dimensions = 2
        xs = torch.linspace(1, 10, steps=100)
        ys = torch.linspace(1, 10, steps=100)
        self.cartesian_space = torch.cartesian_prod(xs, ys)
        print("Created cartesian space of size:", self.cartesian_space.shape)
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter('cartesian_vertices_%d' % i, Parameter(self.cartesian_space, requires_grad=not fixed))

    def forward(self):
        x = 2
    
    def get_cartesian_space(self):
        return self.cartesian_space
    
    def set_cartesian_space(self, cartesian_space):
        self.cartesian_space = cartesian_space

    
class EllipsoidModel(nn.Module):
    def __init__(self, n_output, fix_points=[False]):
        # print("Model init is called")
        super(EllipsoidModel, self).__init__()
        self.ellipseLayer = EllipsoidLayer(fix_points)
    
    def forward(self, x, coeffs_t):
        x = 2
    
    def get_cartesian_space(self):
        return self.ellipseLayer.get_cartesian_space()
    
    def set_cartesian_space(self, cartesian_space):
        return self.ellipseLayer.set_cartesian_space(cartesian_space)

import random
def train_ellipsoid_volume(number_of_random_points, model, criterion, optimizer, vol_reg):
    loss_sum = 0.0
    model.train()

    # Gets a random point n number of times, and if it's not in the ellipse volume
    # Removes it from the cartesian space
    truth_sum = 0
    for _ in range(number_of_random_points):
        acc_loss = 0.
        cartesian_space = model.get_cartesian_space()
        chosen_input = random.choice(cartesian_space)
        # print(chosen_input)
        acc_loss, truth_value = criterion.get_ellipsoid_loss(chosen_input)

        if truth_value == False:
            # print(cartesian_space.tolist()[0])
            chosen_input = chosen_input.tolist()
            cartesian_space_list = cartesian_space.tolist()
            cartesian_space_list.remove(chosen_input)
            cartesian_space = torch.tensor(cartesian_space_list)
        
        model.set_cartesian_space(cartesian_space)
        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()
        loss = torch.tensor(int(truth_value)) - vol_reg * log_vol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * 1
        truth_sum += int(truth_value)

    return {
        'loss': loss_sum/number_of_random_points,
        'accuracy': int(truth_sum)/number_of_random_points,
    }
        
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
        criterion = EllipsoidLossFunction(2)
        columns = ['vert', 'ep', 'lr', 'tr_loss', 'tr_acc', 'time', "vol"]
        for epoch in range(args.epochs):
            time_ep = time.time()
            train_res = train_ellipsoid_volume(10, simplex_model, criterion, optimizer, reg_pars[vv])
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