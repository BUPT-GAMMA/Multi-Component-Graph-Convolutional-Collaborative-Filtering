import torch
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0Dense(nn.Module):
    """
    Implementation of L0 regularization for the input units of a fully connected layer
    """
    def __init__(self, feature, embed_dim, weight_decay = 0.0005, droprate = 0.5, bias = False,
                    temperature = 2. / 3., lamda = 1., local_rep = False, **kwargs):
        """
        feature: input dimension
        embed_dim: output dimension
        bias: whether use a bias
        weight_decay: strength of the L2 penalty
        droprate: dropout rate that the L0 gates will be initialized to
        temperature: temperature of the concrete distribution
        lamda: strength of the L0 penalty
        local_rep: whether use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()

        self.feature = feature
        self.embed_dim = embed_dim
        self.prior_prec = weight_decay
        self.temperature = temperature
        self.droprate = droprate
        self.lamda = lamda
        self.use_bias = bias
        self.local_rep = local_rep

        self.weights = Parameter(torch.Tensor(feature, embed_dim))
        self.qz_loga = Parameter(torch.Tensor(feature))
        if bias:
            self.bias = Parameter(torch.Tensor(embed_dim))

        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode = 'fan_out')
        self.qz_loga.data.normal_(math.log(1 - self.droprate) - math.log(self.droprate), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min = math.log(1e-2), max = math.log(1e2))

    def cdf_qz(self, x):
        # Implements CDF of the stretched concrete distribution
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min = epsilon, max = 1 - epsilon)

    def quantile_concrete(self, x):
        # Implements the quantile of stretched concrete distribution
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        # Expected L0 norm under the stochastic gates
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamda, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def get_eps(self, size):
        # Uniform random numbers for the concrete distribution
        eps = self.floatTensor(size).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample = True):
        # Sample the hard-concrete gates for training and use a deterministic value for testing
        # training
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.feature))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val = 0, max_val = 1)
        # testing
        else:
            pi = torch.sigmoid(self.qz_loga).view(1, self.feature).expand(batch_size, self.feature)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val = 0, max_val = 1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.feature)))
        mask = F.hardtanh(z, min_val = 0, max_val = 1)
        return mask.view(self.feature, 1) * self.weights

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample = self.training)
            xin = input.mul(z)
            output = xin.mm(self.weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)

        if self.use_bias:
            output.add_(self.bias)
        return output