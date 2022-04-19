import numpy as np
import time
import torch
import argparse
from tqdm import tqdm
import math


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='Diginetica2/Tmall/NowPlaying/retailrocket')
parser.add_argument('--freq',type=int,default=20)
parser.add_argument('--cold_ratio',type=float,default=0.1)
parser.add_argument('--seq_ratio',type=float,default=1.0)
parser.add_argument('--device',default='cpu')



opt = parser.parse_args()
print(opt)
# dataset = 'Tmall'seqK
# dataset = 'Diginetica2'
# dataset = 'NowPlaying'
# dataset = 'retailrocket'


'''根据全局图创建三种图. 1)顺序关系图; 2)共现关系图; 3)互斥关系图'''

def BOOL(np_array):
    if str(type(np_array)) == str(type(torch.tensor([1, 2, 3]).to(device))):
        np_array1 = np_array.clone()
    elif str(type(np_array)) == str(type(np.array([1, 2, 3]))):
        np_array1 = np_array.copy()
    else:
        np_array1 = []
        assert 1==2
    np_array1[np_array1 > 1] = 1

    np_array1[np_array1 < 0] = 0

    return np_array1


'1. 创建顺序、共现关系图'

def get_seq_co_graph(g, less_frequency=100, seq_ratio=1.0):
    g_t = g.T
    total_num_ = g + g_t
    gt_less_frequency = total_num_.gt(less_frequency)  # wij+wji > less_freq
    le_less_frequency = ~gt_less_frequency  # wij+wji <= less_freq

    r_co = torch.min(g, g_t)  # co = min{wij, wji} 因为对于wij=0 wji=1的值会赋co_ij=co_ji=0 所以共现图好多0
    r_seq = torch.max(g, g_t) - r_co  # seq = max{wij, wji} - min{wij, wji}

    r_seq_set = torch.max(r_seq, r_seq.T)  # seq_{ij} = max{seq_ij, seq_ji}

    mask = r_seq_set.gt(seq_ratio * r_co)  # seq > co ?

    r_seq_set = r_seq_set * mask * gt_less_frequency  # (seq > co) AND (wij + wji > less_freq)

    '''
    co_graph:
        1. co_ij = co_ji
        2. wij+wji<=less_freq    1) wij=0 or wji=0, but seq_ij!=0, co_ij=seq_ij
                                 2) co_ij=min{wij,wji}
    '''
#     r_co = r_co * total_num_.gt(0) * ~(mask * gt_less_frequency) + r_seq * gt_less_frequency * (g.eq(0) + g_t.eq(0))
    r_co = r_co * total_num_.gt(0) * ~(mask * gt_less_frequency) * gt_less_frequency
    # r_co = r_co * (~mask + le_less_frequency * total_num_.gt(0))# (seq <= co) OR (wij + wji <= less_freq  AND wij+wji>0)
    _co_graph = r_co  # get co_graph

    seq_mask = g.gt(g_t)  # w_ij > w_ji ?
    r_seq = r_seq_set * seq_mask  # w_ij > w_ji

    _seq_graph = r_seq  # get seq_graph

    _is_eq = torch.eq(_seq_graph, _co_graph)
    seq_is_zero_ = torch.masked_select(_seq_graph, _is_eq)
    co_is_zero = torch.masked_select(_co_graph, _is_eq)
    assert seq_is_zero_.sum() == 0
    assert co_is_zero.sum() == 0
    _is_directed = torch.eq(_seq_graph, _seq_graph.T)
    seq_is_zero_ = torch.masked_select(_seq_graph, _is_directed)
    seq_t_is_zero_ = torch.masked_select(_seq_graph.T, _is_directed)
    assert seq_is_zero_.sum() == 0
    assert seq_t_is_zero_.sum() == 0

    return _seq_graph, _co_graph


'2. 创建不相容关系图'
def get_in_graph1(g, cold_item_, less_freq_=100):
    graph_ = g + g.T
    never_co_exist_ = graph_.eq(0)
    graph_ = graph_ - torch.diag(torch.diag(graph_))
    gt_less_freq_ = graph_.gt(less_freq_)
    graph_ = graph_ * gt_less_freq_  # 断掉所有交互次数过少的边

    two_hop_ = graph_ @ graph_
    same_context_ = two_hop_.gt(0)

    is_incompatible_ = never_co_exist_ * same_context_

    # is_incompatible_ = ~is_1_hop_ * is_2_hop_ * ~is_3_hop_
    in_graph_ = 1 * is_incompatible_
    in_graph_ = in_graph_ - torch.diag(torch.diag(in_graph_))  # get in_graph (0,1)matrix

    '不考虑long-tail item的互斥性'
    in_graph_[cold_item_, :] = 0
    in_graph_[:,cold_item_] = 0

    incompatible_strength_graph_ = torch.zeros(graph_.shape)

    for node1 in tqdm(range(graph_.shape[0]), total=graph_.shape[0], ncols=100):  # item a
        node2 = torch.squeeze(torch.nonzero(in_graph_[node1]))  # incompatible item b

        if node2.shape == torch.Size([0]) or node2 == torch.Size([]):
            continue

        node1_neighbor_freq_ = graph_[node1]
        node2_neighbor_freq_ = graph_[node2]

        node1_neighbor_ = node1_neighbor_freq_.gt(0)
        node2_neighbor_ = node2_neighbor_freq_.gt(0)

        is_common_ = node1_neighbor_ * node2_neighbor_  # bridges between item a and b

        if len(is_common_.shape) == 1:
            node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_)
            node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_)
        else:
            node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_, dim=1)
            node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_, dim=1)

        in_strength_ = node1_common_freq_ + node2_common_freq

        incompatible_strength_graph_[node1, node2] = in_strength_

    in_graph_ = in_graph_ * incompatible_strength_graph_  # get incompatible graph with in-strength

    is_undirected_ = torch.eq(in_graph_, in_graph_.T)

    assert is_undirected_.sum() == graph_.shape[0] * graph_.shape[1]

    return in_graph_





def get_in_graph2(g, cold_item):
    print('生成incompatible graph:')
    graph_ = BOOL(g)  # 先把全局图的邻接矩阵变成bool矩阵
    com_out_num = graph_ @ graph_.t()  # ij的共同后继个数
    com_in_num = graph_.t() @ graph_  # ij的共同前驱个数

    g_degree = com_out_num + com_in_num  # ij的公共前驱和后继的总数
    g_in_ = BOOL(com_out_num) * BOOL(com_in_num) - graph_ - graph_.t()  # i与j有前有后，i没连过j，j也没练过i
    g_in_ = BOOL(BOOL(g_in_) - torch.eye(graph_.shape[0]).to(device))
    g_in_ = g_in_ * g_degree
    g_in_[cold_item,:] = 0
    g_in_[:,cold_item] = 0
    return g_in_


'3. 创建三种关系图'
def get_multi_faceted_graphs(global_graph_, freq_matrix_, less_freq_=10, cold_ratio_=0.1, seq_ratio_=1.0):

    freq_sorted_ = torch.sort(freq_matrix_,descending=False)[1]  # indices
    cold_item_ = freq_sorted_[:math.ceil(len(freq_sorted_) * cold_ratio_)]
#     cold_item_ = torch.squeeze(torch.nonzero(freq_matrix_.le(100)))

    in_graph_ = get_in_graph1(global_graph_, cold_item_=cold_item_, less_freq_=less_freq_)  # 基于2阶关系的
    # in_graph_ = get_in_graph2(global_graph_, cold_item=cold_item_)  # 基于相同上下文的

    seq_graph_, co_graph_ = get_seq_co_graph(global_graph_, less_frequency=less_freq_, seq_ratio=seq_ratio_)
    is_incompatible_ = in_graph_.le(0)
    seq_graph_ = seq_graph_ * is_incompatible_
    co_graph_ = co_graph_ * is_incompatible_
    'assert conditions'
    seq_gt_zero_ = seq_graph_.gt(0)
    co_gt_zero_ = co_graph_.gt(0)
    in_gt_zero_ = in_graph_.gt(0)
    # seq != co, seq != in, co != in
    assert torch.sum(seq_gt_zero_ * co_gt_zero_) == 0
    assert torch.sum(seq_gt_zero_ * in_gt_zero_) == 0
    assert torch.sum(co_gt_zero_ * in_gt_zero_) == 0
    # seq is_directed, co and in are is_undirected
    assert torch.sum(torch.eq(in_graph_, in_graph_.T)) == global_graph_.shape[0] * global_graph_.shape[1]
    assert torch.sum(torch.eq(co_graph_, co_graph_.T)) == global_graph_.shape[0] * global_graph_.shape[1]
    assert torch.sum(torch.eq(seq_graph_, seq_graph_.T) * seq_gt_zero_) == 0
    return seq_graph_, co_graph_, in_graph_



def get_edge(g,directed=True):
    all_edges_ = g.gt(0)

    if directed:  # directed graph
        edge_num_ = all_edges_.sum()
    else:  # undirected graph
        self_loop_ = torch.diagonal(all_edges_)
        edge_num_ = (all_edges_.sum()-self_loop_.sum())/2+self_loop_.sum()
    return edge_num_

if __name__ == '__main__':
    dataset = opt.dataset
    device = opt.device
    global_graph_path = '../../dataSet/' + dataset + '/train_test_data/global_graph.npy'
    global_graph = torch.from_numpy(np.load(global_graph_path).astype('float32')).to(device)

    freq_matrix_path = '../../dataSet/' + dataset + '/train_test_data/freq_matrix.npy'
    freq_matrix = torch.from_numpy(np.load(freq_matrix_path).astype('float32')).to(device)

    print(f'global graph shape = {global_graph.shape}')
    print(f'freq matrix shape = {freq_matrix.shape}')

    '顺序、共现图-----------------------------------------'
    with torch.no_grad():
        t11 = time.time()
        print('开始生成图',t11)
        seq_graph,co_graph,in_graph = get_multi_faceted_graphs(global_graph_=global_graph,freq_matrix_=freq_matrix,less_freq_=opt.freq,cold_ratio_=opt.cold_ratio,seq_ratio_=opt.seq_ratio)


        seq_edges = get_edge(seq_graph,directed=True)
        co_edges = get_edge(co_graph,directed=False)
        in_edges = get_edge(in_graph,directed=False)

        t12 = time.time()

        print(f'顺序图边数={seq_edges}')
        print(f'共现图边数={co_edges}')
        print(f'互斥图边数={in_edges}')
        print(f'cost time = {round(t12 - t11, 1)}s')

        seq_graph = seq_graph.cpu().numpy()
        co_graph = co_graph.cpu().numpy()
        in_graph = in_graph.cpu().numpy()

    #
    # assert 1==2
    np.savez(
        '../../dataSet/' + dataset + '/train_test_data/relation_graph3',
        seq_graph=seq_graph,
        co_graph=co_graph,
        in_graph=in_graph
    )

    print('save over~')
