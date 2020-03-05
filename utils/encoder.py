import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class encoder(nn.Module):
    def __init__(self, embedding_dim, aggregator, cuda = "cpu", is_user_part = True):
        super(encoder, self).__init__()

        self.embed_dim = embedding_dim
        self.aggregator = aggregator
        self.device = cuda
        self.is_user = is_user_part

        self.layer = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self, nodes_u, nodes_i):
        # self-connection could be considered
        if self.is_user == True:
            nodes_fea, embed_matrix = self.aggregator(nodes_u)
            combined = torch.cat((nodes_fea, embed_matrix[nodes_u.cpu().numpy()]), dim = 1)
        else:
            nodes_fea, embed_matrix = self.aggregator(nodes_i)
            combined = torch.cat((nodes_fea, embed_matrix[nodes_i.cpu().numpy()]), dim = 1)

        cmp_embed_matrix = self.layer(combined).to(self.device)
        return cmp_embed_matrix