import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class NCF(nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.user_num = config.user_num
        self.conv_num = config.conv_num
        self.factor_dim = config.factor_dim
        self.mlp_factor_dim = config.text_factor_dim
        self.mlp_layers_num = config.mlp_layers_num
        self.mlp_layers = [self.mlp_factor_dim]
        for idx in xrange(self.mlp_layers_num - 1):
            self.mlp_layers.append(self.mlp_layers[idx] / 2)
        self.mlp_layers.append(self.mlp_factor_dim * 2)

        self.mf_user_embedding = nn.Embedding(self.user_num, self.factor_dim)
        nn.init.xavier_normal_(self.mf_user_embedding.weight)
        self.mf_conv_embedding = nn.Embedding(self.conv_num, self.factor_dim)
        nn.init.xavier_normal_(self.mf_conv_embedding.weight)
        self.mlp_user_embedding = nn.Embedding(self.user_num, self.mlp_factor_dim)
        nn.init.xavier_normal_(self.mlp_user_embedding.weight)
        self.mlp_conv_embedding = nn.Embedding(self.conv_num, self.mlp_factor_dim)
        nn.init.xavier_normal_(self.mlp_conv_embedding.weight)
        self.mlps = nn.ModuleList([nn.Linear(self.mlp_layers[idx-1], self.mlp_layers[idx]) for idx in xrange(self.mlp_layers_num)])
        self.out_layer = nn.Linear(self.factor_dim + self.mlp_layers[self.mlp_layers_num-1], 1)
        self.activate = nn.ReLU()
        self.final = nn.Sigmoid()

    def forward(self, uc_pairs):
        if torch.cuda.is_available():  # run in GPU
            uc_pairs = uc_pairs.cuda()
        users = uc_pairs[:, 0]
        convs = uc_pairs[:, 1]
        # MF part
        mf_users_latent = self.mf_user_embedding(users)
        mf_convs_latent = self.mf_conv_embedding(convs)
        mf_vectors = mf_users_latent * mf_convs_latent
        # MLP part
        mlp_users_latent = self.mlp_user_embedding(users)
        mlp_convs_latent = self.mlp_conv_embedding(convs)
        mlp_vectors = torch.cat((mlp_users_latent, mlp_convs_latent), dim=1)
        for idx in xrange(self.mlp_layers_num):
            mlp_vectors = self.activate(self.mlps[idx](mlp_vectors))
        # Final prediction
        final_vectors = torch.cat((mf_vectors, mlp_vectors), dim=1)
        final_outs = self.final(self.out_layer(final_vectors).view(-1))

        return final_outs


