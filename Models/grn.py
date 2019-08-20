import torch
import torch.nn.functional as F
from torch import nn, optim


class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_num):
        super(GRN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_num = state_num
        # input gate, u state, forget gate, output gate
        self.iofu_input_pre = nn.Linear(self.input_dim, 4 * self.hidden_dim)
        self.iofu_input_suc = nn.Linear(self.input_dim, 4 * self.hidden_dim)
        self.iofu_hidden_pre = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        self.iofu_hidden_suc = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        # initial state
        self.initial_hidden_state = nn.Parameter(torch.randn(self.hidden_dim))
        self.initial_cell_state = nn.Parameter(torch.randn(self.hidden_dim))

    def forward(self, conv_rep, arc_in, arc_out):
        batch_size = conv_rep.size(0)
        turn_num = conv_rep.size(1)
        hidden_state = self.initial_hidden_state.repeat(batch_size, turn_num, 1)
        cell_state = self.initial_cell_state.repeat(batch_size, turn_num, 1)
        for t in range(self.state_num):
            # Compute the states
            iofu_pre = self.iofu_input_pre(conv_rep) + self.iofu_hidden_pre(hidden_state)
            iofu_suc = self.iofu_input_suc(conv_rep) + self.iofu_hidden_suc(hidden_state)
            iofu_pre = torch.bmm(arc_in, iofu_pre)
            iofu_suc = torch.bmm(arc_out, iofu_suc)
            iofu = iofu_pre + iofu_suc
            i, o, f, u = torch.split(iofu, self.hidden_dim, dim=-1)
            i, o, f, u = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(f), torch.sigmoid(u)

            cell_state = torch.mul(f, cell_state) + torch.mul(i, u)
            hidden_state = torch.mul(o, torch.tanh(cell_state))

        return hidden_state



