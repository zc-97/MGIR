from train import train
import numpy as np
import torch
from scipy import sparse
import time
import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Gowalla', help='Diginetica2/Tmall/NowPlaying')
parser.add_argument('--embedding_size', type=int, default=100, help='gcegnn100, 128/64')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--dropout_gcn', type=float, default=0.5, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0.2, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--device',  default='cuda:0', help='cuda:1, cpu')

opt = parser.parse_args()

dataset = opt.dataset

# dataset = 'Diginetica'
# dataset = 'Tmall'

path = '../../dataSet/' + dataset + '/train_test_data/'
if dataset == 'Gowalla' or dataset == 'gowalla':
    # graphs1 = np.load(path + 'relation_graph_degree_matrix.npz', allow_pickle=True)  # 一千多万条互斥边
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)

    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 29510
elif dataset == 'Diginetica' or dataset == 'diginetica':
    # graphs1 = np.load(path + 'relation_graph_degree_matrix.npz', allow_pickle=True)  # 一千多万条互斥边
    graphs = np.load(path + 'relation_graph_degree_matrix2.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 42596
elif dataset == 'Lastfm' or dataset == 'lastFM' or dataset == 'LastFM' or dataset == 'LastFm':
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 38615
elif dataset == 'retailrocket':
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 36968
elif dataset == 'Diginetica2' or dataset == 'diginetica2' :
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 43097  # graph 40729 说40729 out
    # opt.dropout_gcn = 0.2
    # # opt.dropout_local = 0.2  # global = local 0.2 还可以
    # opt.dropout_local = 0.4
elif dataset == 'Tmall' or dataset == 'tmall' :
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 40727  # graph 40729 说40729 out
    # opt.dropout_gcn = 0.6
    # opt.dropout_local = 0.5
    opt.dropout_gcn = 0.2
    opt.dropout_local = 0.2
elif dataset == 'NowPlaying':
    graphs = np.load(path + 'relation_graph_degree_matrix3.npz', allow_pickle=True)
    train_test_data = np.load(path + 'train_test_numpy_data.npz', allow_pickle=True)
    item_num = 60416  # graph 40729 说40729 out
    # opt.dropout_gcn = 0.0
    # opt.dropout_local = 0.0

else:
    graphs1 = 0
    graphs = 0
    item_num = 0
    train_test_data = 0

def get_seq_mask_label(data, type):
    seq = torch.from_numpy(data[type + '_seq'])
    mask = torch.from_numpy(data[type + '_mask'])
    label = torch.from_numpy(data[type + '_label'])
    return (seq, mask, label)

def np_pad(np_array):
    right_bottom_pad = np.pad(np_array, ((0, 1), (0, 1)), 'constant', constant_values=0)
    return right_bottom_pad

def get_graphs(item_num_, _graphs):
    seq_ = _graphs['seq_graph']
    co_ = _graphs['co_graph']
    in_ = _graphs['in_graph']
    seq_graph_in = np_pad(seq_[:, :item_num_])
    seq_graph_out = np_pad(seq_[:, item_num_:])
    co_graph_in = np_pad(co_[:, :item_num_])
    co_graph_out = np_pad(co_[:, item_num_:])
    in_graph_in = np_pad(in_[:, :item_num_])
    in_graph_out = np_pad(in_[:, item_num_:])
    seq_graph_in = dense2sparse(seq_graph_in)
    seq_graph_out = dense2sparse(seq_graph_out)
    co_graph_in = dense2sparse(co_graph_in)
    co_graph_out = dense2sparse(co_graph_out)
    in_graph_in = dense2sparse(in_graph_in)
    in_graph_ = torch.from_numpy(in_graph_out)
    in_graph_out = dense2sparse(in_graph_out)

    return seq_graph_in, seq_graph_out, co_graph_in, co_graph_out, in_graph_in, in_graph_out, in_graph_

def dense2sparse(_matrix):
    a_ = sparse.coo_matrix(_matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix

train_data = get_seq_mask_label(train_test_data, 'train')
x = train_data[0]
y = train_data[-1]
print(x[0])
print(y[:5])

test_data = get_seq_mask_label(train_test_data, 'test')
seq_in, seq_out, co_in, co_out, in_in, in_out, graph_in = get_graphs(item_num, graphs)
session_length = len(train_data[0][0])

print(f'处理数据集{dataset}')
print(f'itemNum={item_num}')
print(f'session length = {session_length}')
print(f'graph_in.shape = {seq_in.shape}')
print('---------------------------------------')
print(opt)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

train(
    dataset=dataset,
    trainData=train_data,
    testData=test_data,
    seq_in=seq_in,
    seq_out=seq_out,
    co_in=co_in,
    co_out=co_out,
    in_in=in_in,
    in_out=in_out,
    graph_in=graph_in,
    item_num=item_num,  # Gowalla:29510
    sess_len=session_length,
    embedding_size=opt.embedding_size,
    out_channel=1,
    gnn_layers=opt.layers,
    epoch=opt.epoch,
    lr_decay=opt.lr_dc,
    lr_decay_epoch=opt.lr_dc_step,
    batch_size=opt.batch_size,
    learning_rate=opt.lr,
    gnn_drop=opt.dropout_gcn,
    local_drop=opt.dropout_local,
    weight_decay=opt.l2,
    alpha=opt.alpha,
    device=opt.device
)
    # 4 下是46 第四下的lr是*了0.1
    # 52.66 26
    # 52.89 26.02  epoch:50
