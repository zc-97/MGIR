import pickle
import numpy as np
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='Diginetica2', help='Diginetica2/Tmall/NowPlaying/retailrocket')
parser.add_argument('--dataset', default='Tmall', help='Diginetica2/Tmall/NowPlaying/retailrocket')
parser.add_argument('--va', default=0, type=int, help='Diginetica2/Tmall/NowPlaying/retailrocket')
opt = parser.parse_args()
print(opt)
# dataset = 'Tmall'
# dataset = 'Diginetica2'
# dataset = 'NowPlaying'
# dataset = 'retailrocket'
dataset = opt.dataset
if dataset == 'Delicious':
    with open(r'../../dataSet/'+dataset+'/original_data/test.txt', 'rb+') as testf:
        test = pickle.load(testf)
    with open(r'../../dataSet/'+dataset+'/original_data/train.txt','rb+') as tr:
        train = pickle.load(tr)
    with open(r'../../dataSet/'+dataset+'/original_data/valid.txt', 'rb') as validf:
        valid = pickle.load(validf)
else:
    # with open(r'../../dataSet/NowPlaying/original_data/test.txt','rb+') as testf:
    with open(r'../../dataSet/'+dataset+'/original_data/test.txt', 'rb+') as testf:
        test = pickle.load(testf)
    with open(r'../../dataSet/'+dataset+'/original_data/all_train_seq.txt','rb+') as tr:
        all_tr_seq = pickle.load(tr)
    with open(r'../../dataSet/'+dataset+'/original_data/train.txt','rb+') as tr:
        train = pickle.load(tr)
    all = 0

    for seq in all_tr_seq:
        all += len(seq)
    for seq in test[0]:
        all += len(seq)
    for seq in train[0]:
        all += len(seq)
    print('avg length: ', all/(len(all_tr_seq) + len(train[0]) +len(test[0]) * 1.0))
    print('all:', all)
    # assert 1==2
    def claim_sess_length(tr_,te_):
        tr_num = 0
        te_num = 0

        _session_length_list = []

        for sess in tr_[0] + te_[0]:
            _session_length_list.append(len(sess))

        total_num = len(tr_) + len(te_)
        average_length = sum(_session_length_list) / len(_session_length_list)
        return total_num, average_length

    print(len(train))
    print(len(test))
    totoal_num_, average_length_ = claim_sess_length(train, test)
    print('totoal_num_', totoal_num_)
    print('average_length', average_length_)
    print(len(train[0]))
    print(len(test[0]))
    tr_len = 0
    for sess in train[0]:
        tr_len+=1
    te_len = 0
    for sess in test[0]:
        te_len+=1
    print('Train Session No.',tr_len)
    print('Test Session No.', te_len)
    tr_clicks = 0
    for sess in train[0]:
        for i in sess:
            tr_clicks += 1
    te_clicks = 0
    for sess in test[0]:
        for i in sess:
            te_clicks += 1
    trs_clicks = 0
    trs_len = 0
    for sess in all_tr_seq:
        trs_len += 1
        for i in sess[:-1]:
            trs_clicks+=1
    print('Average Length of Train Sessions', tr_clicks/tr_len)
    print('Average Length of Test Sessions.', te_clicks / te_len)
    print('Average length of all tr seq :', trs_clicks / trs_len)
    print('ave tr+te = ', (tr_clicks+te_clicks) / (tr_len+te_len) )
    print('ave tr_trs = ', (tr_clicks + trs_clicks) / (tr_len + trs_len))
    print('ave te+trs =', (te_clicks + trs_clicks) / (te_len + trs_len))
    print('ave all = ', (tr_clicks + trs_clicks + te_clicks) / (tr_len + trs_len + te_len))

    print(len(all_tr_seq))
    print(train[0][:5])
    print(train[1][:5])
    print(all_tr_seq[:2])
    original_sess_num = len(all_tr_seq)
    clicks = 0
    for sess in all_tr_seq:
        for i in sess:
            clicks += 1
    print('clicks = ', clicks)

    aux_sess_num = len(train[0])+len(test[0])


    def minus_item_id4train_seq(train_seq_):
        new_train_seq_ = []
        for sess in train_seq_:
            new_sess_ = []
            for item in sess:
                new_item = item-1
                if item == 40728:
                    print(item)
                    print(new_item)
                    assert 1==2
                new_sess_.append(new_item)
            new_train_seq_.append(new_sess_)
        return new_train_seq_


    def minus_item_id(seq):
        x = seq[0]
        y = seq[1]
        new_x = []
        new_y = [item-1 for item in y]
        for sess in x:
            new_sess = []
            for item in sess:
                new_sess.append(item-1)
            new_x.append(new_sess)
        return [new_x] + [new_y]


    def get_padding_item(tr_seq_):
        x_ = tr_seq_[0]
        y_ = tr_seq_[1]
        item_list_ = set([])
        for sess in x_:
            for item in sess:
                item_list_.add(item)
        for item in y_:
            item_list_.add(item)

        item_list_ = list(item_list_)
        item_num_ = len(item_list_)
        print('max', max(item_list_))
        print('min', min(item_list_))

        padding_item_id_ = max(item_list_) + 1

        return item_num_, padding_item_id_


    def create_global_matrix(all_train_seq, item_num):

        _matrix = np.zeros((item_num, item_num), dtype=int)
        _freq_matrix = np.zeros(item_num,dtype=int)
        # print(_freq_matrix.shape)
        for sess in all_train_seq:

            # 遍历第 1 ~ n-1 个item
            for i in range(len(sess) - 1):
                _freq_matrix[sess[i]] += 1
                # 遍历从 i+1 ~ n 个item
                for j in range(i + 1, len(sess)):
                    assert j < len(sess)

                    item_i = sess[i]

                    item_j = sess[j]

                    # aij 表示 i之后出现j的次数
                    _matrix[item_i, item_j] += 1
                    if i == (len(sess) - 1):
                        _freq_matrix[sess[j]] += 1

        return _matrix,_freq_matrix


    def padding_sessions(seq_, padding_item_id_, max_length_):
        seq_x = seq_[0]
        label_seq_ = seq_[1]
        # max_length_ = 0
        # for sess in seq_x:
        #     length_ = len(sess)
        #     if length_ > max_length_:
        #         max_length_ = length_
        new_seq_ = []
        new_mask_ = []
        for sess in seq_x:
            masked_sess_ = [padding_item_id_] * (max_length_ - len(sess)) + sess
            mask_ = [0] * (max_length_ - len(sess)) + [1] * len(sess)
            new_seq_.append(masked_sess_)
            new_mask_.append(mask_)
        print('max_len=',max_length_)
        return new_seq_, new_mask_, label_seq_

    def get_train_seq(tr_):
        x_ = tr_[0]
        y_ = tr_[1]
        new_seq_ = []
        for i_ in range(len(x_)):
            sess_ = x_[i_]
            sess_tail = y_[i_]
            new_sess_ = sess_ + [sess_tail]
            new_seq_.append(new_sess_)
        return new_seq_

    if __name__ == '__main__':



        train = minus_item_id(train)
        test = minus_item_id(test)
        all_tr_seq = get_train_seq(train)

       # clicks= 2624140

        item_num, padding_item_id = get_padding_item(train)
        print('item_num=',item_num)
        print('padding index',padding_item_id)
        global_matrix,freq_matrix = create_global_matrix(all_tr_seq, item_num)
        click=0
        for sess in all_tr_seq:
            for i in sess:
                click += 1
        print('click',click)
        print('ave',click/40727)
        print('avg.2', click/len(all_tr_seq))
        click=0
        for sess in all_tr_seq:
            for i in sess[:-1]:
                click += 1
        print('avg.3', click/len(all_tr_seq))

        # assert 1==2
        sess_num = 0
        trx = train[0]
        tex = test[0]

        sess_len = 0
        for sess in trx:
            sess_len += len(sess)
        for sess in tex:
            sess_len += len(sess)
        print('sess len',sess_len)
        print('average sess len,', (sess_len+len(trx)+len(tex))/(len(trx)+len(tex)))

        x = test[0]
        y = test[1]
        for sess in x:
            sess_num += 1
            for i in sess:
                click +=1
        print('click', click)
        print('sess num', sess_num)
        for i in y:
            click += 1
        print('click,', click)
        max_len = 0
        for sess in train[0]:
            if len(sess) >= max_len:
                max_len = len(sess)
        for sess in test[0]:
            if len(sess) >= max_len:
                max_len = len(sess)
        print(max_len)
        if dataset == 'Diginetica2':
            assert max_len == 69
        new_train = padding_sessions(train, padding_item_id, max_length_=max_len)

        # 在这测试tr和va是不是一回事
        print('在这测试tr和va是不是一回事')
        print(len(new_train))
        print(len(new_train[0]))
        print(new_train[0][0])
        print(new_train[1][0])
        print(new_train[2][0])



        def split_validation(tr_seq,valid_portion):
            tr_seq_x = tr_seq[0]
            tr_seq_mask = tr_seq[1]
            tr_seq_y = tr_seq[2]
            n_samples = len(tr_seq[0])
            sidx = np.arange(n_samples,dtype='int32')
            np.random.shuffle(sidx)
            n_train = int(np.round(n_samples * (1. - valid_portion)))

            new_tr_seq_x = [tr_seq_x[s] for s in sidx[:n_train]]
            new_tr_seq_mask = [tr_seq_mask[s] for s in sidx[:n_train]]
            new_tr_seq_y = [tr_seq_y[s] for s in sidx[:n_train]]

            new_va_seq_x = [tr_seq_x[s] for s in sidx[n_train:]]
            new_va_seq_mask = [tr_seq_mask[s] for s in sidx[n_train:]]
            new_va_seq_y = [tr_seq_y[s] for s in sidx[n_train:]]

            new_tr_seq = (new_tr_seq_x, new_tr_seq_mask, new_tr_seq_y)
            new_va_seq = (new_va_seq_x, new_va_seq_mask, new_va_seq_y)

            return new_tr_seq,new_va_seq


        if opt.va==1:
            print('切分验证集')
            new_train,new_test = split_validation(new_train,0.1)
        else:
            new_test = padding_sessions(test, padding_item_id, max_length_=max_len)

        sess_num = len(new_train[1]+new_test[1])
        average_sess_len = [len(sess) for sess in train[0]] + [len(sess) for sess in test[0]]
        average_sess_len = sum(average_sess_len) / sess_num

        print('sess_num', sess_num)
        print('avg. sess len', average_sess_len)
        print('x',new_train[0][0])
        print('mask',new_train[1][0])
        print('y',new_train[2][0])
        print('te x', new_test[0][0])
        print('te mask', new_test[1][0])
        print('te y', new_test[2][0])
        # assert 1==2
        save_path = '../../dataSet/'+dataset+'/train_test_data/'
        with open(save_path + 'train_seq.txt', 'wb') as train_txt:
            pickle.dump(new_train, train_txt)
        with open(save_path + 'test_seq.txt', 'wb') as test_txt:
            pickle.dump(new_test, test_txt)
        np.save(save_path + 'global_graph', global_matrix, allow_pickle=True)
        np.save(save_path + 'freq_matrix',freq_matrix,allow_pickle=True)
        print('save over')

