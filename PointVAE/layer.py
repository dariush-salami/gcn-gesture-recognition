from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch
from torch_geometric.nn.inits import reset



class MuConv(MessagePassing):
    def __init__(self, local_nn=None, global_nn=None, **kwargs):
        super(MuConv, self).__init__(aggr='add', **kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn

        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, edge_index):
        # Add self-loops for symmetric adjacencies.
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        msg = x_j
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def update(self, aggr_out):
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)


class SigmaConv(MessagePassing):
    def __init__(self, local_nn=None, global_nn=None, **kwargs):
        super(SigmaConv, self).__init__(aggr='add', **kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn

        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, edge_index):
        # Add self-loops for symmetric adjacencies.
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):

        msg = x_j - x_i
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def update(self, aggr_out):
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)