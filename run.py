# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from utils.l0dense import L0Dense
from utils.encoder import encoder
from utils.combiner import combiner
from torch.autograd import Variable
from utils.aggregator import aggregator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



class MCCF(nn.Module):
    def __init__(self, u_embedding, i_embedding, embed_dim, N = 30000, droprate = 0.5, beta_ema = 0.999):
        super(MCCF, self).__init__()

        self.u_embed = u_embedding
        self.i_embed = i_embedding
        self.embed_dim = embed_dim
        self.N = N
        self.droprate = droprate
        self.beta_ema = beta_ema

        self.u_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.u_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.i_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.i_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.ui_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ui_layer2 = nn.Linear(self.embed_dim, 1)

        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)
        self.i_bn = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)
        self.ui_bn = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        nodes_u_embed = self.u_embed(nodes_u, nodes_i)
        nodes_i_embed = self.i_embed(nodes_u, nodes_i)

        x_u = F.relu(self.u_bn(self.u_layer1(nodes_u_embed)), inplace = True)
        x_u = F.dropout(x_u, training = self.training, p = self.droprate)
        x_u = self.u_layer2(x_u)

        x_i = F.relu(self.i_bn(self.i_layer1(nodes_i_embed)), inplace = True)
        x_i = F.dropout(x_i, training = self.training, p = self.droprate)
        x_i = self.i_layer2(x_i)

        x_ui = torch.cat((x_u, x_i), dim = 1)
        x = F.relu(self.ui_bn(self.ui_layer1(x_ui)), inplace = True)
        x = F.dropout(x, training = self.training, p = self.droprate)

        scores = self.ui_layer2(x)
        return scores.squeeze()

    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)
        loss = self.criterion(scores, ratings)

        total_loss = loss + self.regularization()
        return total_loss


def train(model, train_loader, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_u, batch_i, batch_ratings = data

        optimizer.zero_grad()
        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device))
        loss.backward(retain_graph = True)
        optimizer.step()

        avg_loss += loss.item()

        # clamp the parameters
        layers = model.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        if model.beta_ema > 0.:
            model.update_ema()

        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10, rmse_mn, mae_mn))
            avg_loss = 0.0
    return 0


def test(model, test_loader, device):
    model.eval()

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    pred = []
    ground_truth = []

    for test_u, test_i, test_ratings in test_loader:
        test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
        scores = model(test_u, test_i)
        pred.append(list(scores.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))

    pred = np.array(sum(pred, []), dtype = np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)

    rmse = sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    if model.beta_ema > 0:
        model.load_params(old_params)
    return rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'MCCF')
    parser.add_argument('--epochs', type = int, default = 300,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 64,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'yelp',
                        metavar = 'STRING', help = 'dataset')
    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    data_path = './datasets/' + args.dataset

    with open(data_path + '/_allData.p', 'rb') as meta:
        u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)

    """
    u_adj: user's purchased history (item set in training set)
    i_adj: user set (in training set) who have interacted with the item
    u_train, i_train, r_train: training set (user, item, rating)
    u_test, i_test, r_test: testing set (user, item, rating)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
    											torch.FloatTensor(r_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
    											torch.FloatTensor(r_test))

    _train = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True,
                                            num_workers = 16, pin_memory = True)
    _test = torch.utils.data.DataLoader(testset, batch_size = args.test_batch_size, shuffle = True, 
    										num_workers = 16, pin_memory = True)

    # user part
    u_agg_embed_cmp1 = aggregator(u2e.to(device), i2e.to(device), u_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate)
    u_embed_cmp1 = encoder(embed_dim, u_agg_embed_cmp1, cuda = device)

    u_agg_embed_cmp2 = aggregator(u2e.to(device), i2e.to(device), u_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate)
    u_embed_cmp2 = encoder(embed_dim, u_agg_embed_cmp2, cuda = device)

    u_agg_embed_cmp3 = aggregator(u2e.to(device), i2e.to(device), u_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate)
    u_embed_cmp3 = encoder(embed_dim, u_agg_embed_cmp3, cuda = device)

    u_embed = combiner(u_embed_cmp1, u_embed_cmp2, u_embed_cmp3, embed_dim, args.droprate, cuda = device)

    # item part
    i_agg_embed_cmp1 = aggregator(u2e.to(device), i2e.to(device), i_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate, is_user_part = False)
    i_embed_cmp1 = encoder(embed_dim, i_agg_embed_cmp1, cuda = device, is_user_part = False)

    i_agg_embed_cmp2 = aggregator(u2e.to(device), i2e.to(device), i_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate, is_user_part = False)
    i_embed_cmp2 = encoder(embed_dim, i_agg_embed_cmp2, cuda = device, is_user_part = False)

    i_agg_embed_cmp3 = aggregator(u2e.to(device), i2e.to(device), i_adj, embed_dim, cuda = device,
    							weight_decay = args.weight_decay, droprate = args.droprate, is_user_part = False)
    i_embed_cmp3 = encoder(embed_dim, i_agg_embed_cmp3, cuda = device, is_user_part = False)

    i_embed = combiner(i_embed_cmp1, i_embed_cmp2, i_embed_cmp3, embed_dim, args.droprate, cuda = device)

    # model
    model = MCCF(u_embed, i_embed, embed_dim, args.N, droprate = args.droprate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    rmse_mn = np.inf
    mae_mn = np.inf
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, epoch, rmse_mn, mae_mn, device)
        # ====================     test       ====================
        rmse, mae = test(model, _test, device)

        if rmse_mn > rmse:
            rmse_mn = rmse
            mae_mn = mae
            endure_count = 0
        else:
            endure_count += 1

        print("<Test> RMSE: %.5f, MAE: %.5f " % (rmse, mae))

        if endure_count > 30:
            break

    print('The best RMSE/MAE: %.5f / %.5f' % (rmse_mn, mae_mn))


if __name__ == "__main__":
    main()
