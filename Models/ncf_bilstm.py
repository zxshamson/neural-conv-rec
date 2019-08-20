import torch
import time
import torch.nn.functional as F
from torch import nn, optim
from cnnencoder import CNNEncoder
import torch.nn.utils.rnn as rnn_utils


class NCFBiLSTM(nn.Module):
    def __init__(self, config, conv_data, pretrained_weight=None, ncf_weights=None):
        super(NCFBiLSTM, self).__init__()
        self.user_num = config.user_num
        self.conv_num = config.conv_num
        self.vocab_num = config.vocab_num
        self.mf_factor_dim = config.factor_dim  # 'replying factors' in paper
        self.text_factor_dim = config.text_factor_dim  # 'conversation interaction' factors in paper
        self.kernal_num = config.kernal_num  # kernal number for CNN encoder
        self.bi_direction = config.bi_direction
        self.embed_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.mlp_layers_num = config.mlp_layers_num
        self.conv_data = conv_data
        # matrix factorization factors
        if self.mf_factor_dim:
            self.mf_user_embedding = nn.Embedding(self.user_num, self.mf_factor_dim)
            self.mf_conv_embedding = nn.Embedding(self.conv_num, self.mf_factor_dim)
            if ncf_weights is not None:
                self.mf_user_embedding.load_state_dict({'weight': ncf_weights[0]})
                self.mf_conv_embedding.load_state_dict({'weight': ncf_weights[1]})
            else:
                nn.init.xavier_normal_(self.mf_user_embedding.weight)
                nn.init.xavier_normal_(self.mf_conv_embedding.weight)
        # user text factors
        self.user_embedding = nn.Embedding(self.user_num, self.text_factor_dim)
        if ncf_weights is not None:
            self.user_embedding.load_state_dict({'weight': ncf_weights[2]})
        else:
            nn.init.xavier_normal_(self.user_embedding.weight)
        # word embedding layer and modeling layer
        self.word_embedding = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=0)
        if pretrained_weight is not None:
            self.word_embedding.load_state_dict({'weight': pretrained_weight})
        # turn modeling layer
        self.turn_modeling = CNNEncoder(self.embed_dim, self.kernal_num, config.dropout)
        turn_hidden_dim = self.kernal_num * 3
        self.conv_modeling = nn.LSTM(turn_hidden_dim + self.text_factor_dim,
                                     self.hidden_dim // 2 if self.bi_direction else self.hidden_dim,
                                     bidirectional=self.bi_direction, batch_first=True)
        # conv text hidden states to factors
        self.h2f = nn.Linear(self.hidden_dim, self.text_factor_dim)
        # mlp layers and out layer
        if self.mlp_layers_num:
            self.mlp_layers = [self.text_factor_dim]
            for idx in xrange(self.mlp_layers_num - 1):
                self.mlp_layers.append(self.mlp_layers[idx] / 2)
            self.mlp_layers.append(self.text_factor_dim * 2)
            self.mlps = nn.ModuleList(
                [nn.Linear(self.mlp_layers[idx - 1], self.mlp_layers[idx]) for idx in xrange(self.mlp_layers_num)])
            self.out_layer = nn.Linear(self.mf_factor_dim + self.mlp_layers[self.mlp_layers_num - 1], 1)
        else:
            self.out_layer = nn.Linear(self.mf_factor_dim + self.text_factor_dim * 2, 1)
        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, uc_pairs):
        userids = uc_pairs[:, 0]
        convids = uc_pairs[:, 1]
        convs = self.conv_data[convids]
        if torch.cuda.is_available():  # run in GPU
            userids = userids.cuda()
            convids = convids.cuda()
            convs = convs.cuda()
        # MF part
        if self.mf_factor_dim:
            mf_users_latent = self.mf_user_embedding(userids)
            mf_convs_latent = self.mf_conv_embedding(convids)
            mf_vectors = mf_users_latent * mf_convs_latent
            # print mf_vectors
        # Conv content part
        users_latent = self.user_embedding(userids)

        batch_size = convs.size(0)
        turn_num = (convs[:, :, 4] > 0).sum(dim=1)
        all_turn = torch.cat([convs[i, :turn_num[i], 4:] for i in range(batch_size)], dim=0)
        all_turn_uids = torch.cat([convs[i, :turn_num[i], 0] for i in range(batch_size)], dim=0)
        all_turn_reps = self.turn_modeling(self.word_embedding(all_turn))
        all_turn_reps = torch.cat([self.user_embedding(all_turn_uids), all_turn_reps], dim=1)
        conv_turn_reps = torch.split(all_turn_reps, list(turn_num))

        sorted_turn_num, sorted_indices = torch.sort(turn_num, descending=True)
        _, desorted_indices = torch.sort(sorted_indices, descending=False)
        sorted_conv_reps = []
        for index in sorted_indices:
            sorted_conv_reps.append(conv_turn_reps[index])
        paded_conv_reps = rnn_utils.pad_sequence(sorted_conv_reps, batch_first=True)
        packed_conv_reps = rnn_utils.pack_padded_sequence(paded_conv_reps, sorted_turn_num, batch_first=True)
        conv_out, conv_hidden = self.conv_modeling(packed_conv_reps)
        conv_out = rnn_utils.pad_packed_sequence(conv_out, batch_first=True)[0]
        conv_out = conv_out[desorted_indices]
        if self.bi_direction:
            conv_reps = torch.cat([conv_hidden[0][-2], conv_hidden[0][-1]], dim=1)
            conv_reps = conv_reps[desorted_indices]
        else:
            conv_reps = conv_hidden[0][-1][desorted_indices]

        convs_latent = self.tanh(self.h2f(conv_reps))
        # MLP part
        text_vectors = torch.cat((users_latent, convs_latent), dim=1)
        if self.mlp_layers_num:
            for idx in xrange(self.mlp_layers_num):
                text_vectors = self.relu(self.mlps[idx](text_vectors))
        # Final prediction
        if self.mf_factor_dim:
            final_vectors = torch.cat((mf_vectors, text_vectors), dim=1)
        else:
            final_vectors = text_vectors
        final_outs = torch.sigmoid(self.out_layer(final_vectors).view(-1))

        return final_outs

