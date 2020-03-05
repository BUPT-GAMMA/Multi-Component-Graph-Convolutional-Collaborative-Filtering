import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class attention(nn.Module):
    def __init__(self, embedding_dim, droprate, cuda = "cpu"):
        super(attention, self).__init__()

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax()

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(n_neighs, 1)

        x = torch.cat((feature1, feature2_reps), 1)
        x = F.relu(self.att1(x).to(self.device), inplace = True)
        x = F.dropout(x, training =self.training, p = self.droprate)
        x = self.att2(x).to(self.device)

        att = F.softmax(x, dim = 0)
        return att