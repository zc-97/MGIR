import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Diginetica2', help='Diginetica2/Tmall/NowPlaying/retailrocket')
opt = parser.parse_args()
print(opt)
def bool_numpy(numpy_array):
    numpy_array_1 = numpy_array.copy()
    numpy_array_1[numpy_array_1 == 0] = 0.00001
    return numpy_array_1

# def get_degree_matrix(adj_matrix):
#     d = np.shape(adj_matrix)[0]
#     row = bool_numpy(np.sum(adj_matrix, axis=0))
#     row = np.reshape(row, (1, d))
#     col = bool_numpy(np.sum(adj_matrix, axis=1))
#     col = np.reshape(col, (d, 1))
#     a_in = adj_matrix / col
#     a_out = adj_matrix / row
#     # degree_matrix = np.concatenate((a_in, a_out), axis=1)
#     degree_matrix = np.concatenate((a_in, a_out.T), axis=1)
#     return degree_matrix

def get_degree_matrix(adj_matrix):
    d = np.shape(adj_matrix)[0]
    row = np.sum(adj_matrix,axis=0)
    row = np.reshape(row, (1, d))
    col = np.sum(adj_matrix,axis=1)
    col = np.reshape(col, (d, 1))

    row_zero_ = row.__eq__(0)
    col_zero_ = col.__eq__(0)
    row[row_zero_]=1
    col[col_zero_]=1
    a_in = adj_matrix / col
    a_out = adj_matrix / row
    degree_matrix = np.concatenate((a_in, a_out.T), axis=1)
    return degree_matrix


if __name__ == '__main__':
    # dataSet = 'Tmall'
    # dataSet = 'Diginetica2'
    # dataSet = 'NowPlaying'
    # dataSet = 'retailrocket'
    dataSet = opt.dataset
    path = '../../dataSet/'+dataSet+'/train_test_data/relation_graph3.npz'
    relation_graphs = np.load(path)
    print(f'loading {dataSet} relation graphs.npz')
    seq_graph = relation_graphs['seq_graph']
    co_graph = relation_graphs['co_graph']
    in_graph = relation_graphs['in_graph']

    print('Creating degree matrix ...')

    seq_graph_degree = get_degree_matrix(seq_graph)
    co_graph_degree = get_degree_matrix(co_graph)
    in_graph_degree = get_degree_matrix(in_graph)

    print('Start to save three degree matrix...')
    np.savez(
        '../../dataSet/' + dataSet + '/train_test_data/relation_graph_degree_matrix3',
        seq_graph=seq_graph_degree,
        co_graph=co_graph_degree,
        in_graph=in_graph_degree
    )
    print('~ over ~')
