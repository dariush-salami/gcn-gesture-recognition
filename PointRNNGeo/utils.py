
import scipy.sparse as sp
import torch
import h5py
import numpy as np


def load_data_ply():
    with h5py.File('data/ply_data_train0.h5', 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        data_key = list(f.keys())[0]
        label_key = list(f.keys())[1]
        # Get the data
        data = np.array(f[data_key])
        labels = np.array(f[label_key])
        return data,labels

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    # b = xyz1.shape[0]
    # n = xyz1.shape[1]
    # c = xyz1.shape[2]
    # m = xyz2.shape[1]
    # xyz1 = xyz1.reshape((b, 1, n, c))
    # xyz2 = xyz2.reshape((b, m, 1, c))
    # xyz1 = tile(xyz1,1,m)
    # xyz2 = tile(xyz2, 2, n)
    #xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    #xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    a = xyz1[:,:, None, ...]
    b = xyz2[:,None, ...]
    adj = a - b
    adj = adj.pow(2).sum(-1)
    #dist = torch.sum((xyz1 - xyz2) ** 2,dim=-1)
    #dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)
    val, idx = adj.topk(k=k, dim=-1, largest=False)
    #outi, out = select_top_k(k, dist)
    #idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    #val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    # val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU


    return val, idx

def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)



if __name__=='__main__':
    knn = True
    import numpy as np
    import time

    np.random.seed(100)
    pts = np.random.random((32, 512, 64)).astype('float32')
    tmp1 = np.random.random((1000,8, 100, 3)).astype('float32')
    tmp2 = np.random.random((1000,8, 100, 3)).astype('float32')

    points = torch.tensor(pts)

    radius = 0.1
    nsample = 64
    xyz1 = tmp1[:, 1].reshape([-1, tmp1.shape[2], tmp1.shape[3]])
    xyz2 = tmp1[:, 0].reshape([-1, tmp1.shape[2], tmp1.shape[3]])
    xyz1 = torch.tensor(xyz1)
    xyz2 = torch.tensor(xyz2)
    _, idx = knn_point(nsample, xyz2, xyz1)
    #grouped_points = group_point(points, idx)
