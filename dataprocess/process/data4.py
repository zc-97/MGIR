import numpy as np
import torch
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Diginetica2', help='Diginetica2/Tmall/NowPlaying/retailrocket')
opt = parser.parse_args()
print(opt)

def convert_list_2_array(data):
    seq = np.array(data[0])
    mask = np.array(data[1])
    label = np.array(data[2])
    return seq, mask, label


if __name__ == '__main__':
    # dataSet = 'Tmall'
    # dataSet = 'Diginetica2'
    # dataSet = 'NowPlaying'
    # dataSet = 'retailrocket'
    dataSet = opt.dataset
    path = '../../dataSet/' + dataSet + '/train_test_data/'
    with open(path + 'train_seq.txt', 'rb+') as f1:
        train_data = pickle.load(f1)
    with open(path + 'test_seq.txt', 'rb+') as f2:
        test_data = pickle.load(f2)

    '''seq, mask, label'''
    train_seq, train_mask, train_label = convert_list_2_array(train_data)
    test_seq, test_mask, test_label = convert_list_2_array(test_data)
    print('te x',test_seq[0])
    print('mask',test_mask[0])
    print('y',test_label[0])
    np.savez(
        path+'train_test_numpy_data',
        train_seq=train_seq,
        train_mask=train_mask,
        train_label=train_label,
        test_seq=test_seq,
        test_mask=test_mask,
        test_label=test_label
             )

    print('save over~')