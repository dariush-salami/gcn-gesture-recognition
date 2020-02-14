import torch.nn as nn
import torch.nn.functional as F
from layer import EdgeConv
from utils import knn_point
import torch
from torch.nn import Sequential as Seq,Linear as Lin,ReLU , BatchNorm1d as BN , Dropout

from torch_geometric.nn import global_max_pool


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
class EdgeRNNCell(nn.Module):
    def __init__(self,in_channel,n_hids,time_steps,k,numClasses):
        super(EdgeRNNCell,self).__init__()
        self.in_channel = in_channel
        self.time_steps = time_steps
        self.k = k
        self.edgeConv = EdgeConv(in_channel,n_hids)
        self.mlp = Seq(
            MLP([(self.time_steps-1)*n_hids, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, numClasses))
        #for i in range(time_steps-1):
        #    self.edgeConvlist.append(EdgeConv(in_channel,n_hids))
        #self.net = nn.Sequential(*self.edgeConvlist)

    def forward(self,x,batch = None):
        s_t = torch.tensor([])
        for i in range(1,self.time_steps):
            xyz1 = x[:,i].reshape([-1,x.shape[2],x.shape[3]])
            xyz2 = x[:,i-1].reshape([-1,x.shape[2],x.shape[3]])
            _,idx = knn_point(self.k, xyz1,xyz2)
            row = torch.arange(xyz1.shape[1]).repeat_interleave(self.k)[None,:].repeat(xyz1.shape[0],1)
            row = row + torch.arange(0,xyz1.shape[0]*2048,2048)[None,:].transpose(0,1)
            #print(torch.arange(1024, xyz1.shape[0] * 2048, 2048)[None, :].transpose(0, 1).shape)
            col = idx.reshape((-1,idx.shape[1]*idx.shape[2]))+torch.arange(1024, xyz1.shape[0] * 2048, 2048)[None, :].transpose(0, 1)
            # tmp = idx.to_dense()
            feats = torch.cat((xyz1, xyz2), dim=1).reshape((2*xyz1.shape[0]*xyz1.shape[1],3))
            edge_idx = torch.stack((row.flatten(),col.flatten()),dim=0)
            out = self.edgeConv(feats, edge_idx)
            out = global_max_pool(out, batch)
            s_t = torch.cat((s_t,out),dim=1)
        tmp = self.mlp(s_t)
        return F.log_softmax(tmp, dim=1)