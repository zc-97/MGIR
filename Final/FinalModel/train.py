import numpy as np
import pickle
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from model import MRGSR
import time
from scipy import sparse

# random.seed(1)
# np.random.seed(1)
# np.random.seed(2)
# torch.manual_seed(2)
# torch.cuda.manual_seed_all(2)
# torch.cuda.manual_seed_all(2)
# random.seed(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# device = torch.device('cpu' if torch.cuda.is_available() else 'cuda:1')


def data_loader(data, batchSize=32, shuffle=True, num_works=4, pin=True):
    seq, mask, label = data
    train_db = torch.utils.data.TensorDataset(seq, mask, label)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batchSize, shuffle=shuffle, num_workers=num_works,pin_memory=pin)
    return train_loader

def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def train(dataset, trainData, testData, seq_in, seq_out, co_in, co_out, in_in, in_out,graph_in, item_num=29510,sess_len=0,out_channel=3,
          embedding_size=100, gnn_layers=2, lr_decay=0.1, lr_decay_epoch=4,
          epoch=300, batch_size=32, learning_rate=0.001, gnn_drop=0.0, local_drop=0.0, weight_decay=0, alpha=0,device='cpu',print_cost=True):
    train_loader = data_loader(trainData, batchSize=batch_size)
    best_result = [0] * 4
    _best_result = [0] * 4
    print('创建网络结构')
    # 创建网络结构
    init_seed(2021)
    seq_in = seq_in.to(device)
    seq_out = seq_out.to(device)
    co_in = co_in.to(device)
    co_out = co_out.to(device)
    in_in = in_in.to(device)
    in_out = in_out.to(device)
#     graph_in = graph_in.to(device)


    test_loader = data_loader(testData, batchSize=batch_size, shuffle=False)
    net = MRGSR(item_num=item_num,
                session_len=sess_len,
                embedding_size=embedding_size,
                gnn_layers=gnn_layers,
                out_channel=out_channel,
                gnn_drop=gnn_drop,
                local_drop=local_drop,
                seq_in=seq_in,
                seq_out=seq_out,
                co_in=co_in,
                co_out=co_out,
                in_in=in_in,
                in_out=in_out,
#                 graph_in=graph_in,
                lr=learning_rate,
                w_decay=weight_decay,
                lr_dc=lr_decay,
                lr_dc_step=lr_decay_epoch,
                alpha=alpha,
                device=device
                ).to(device)


    # cost_fuc = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay)
    costs = []

    epoch_num = 0
    for epoch in tqdm(range(epoch), total=epoch, ncols=100):
        epoch_num += 1
        epoch_cost = 0
        '''训练'''
        net.train()

        # if epoch_num % lr_decay_epoch == 0 and epoch_num != 0:
        #     learning_rate = learning_rate * lr_decay

        batch = 0
        for step, data in enumerate(train_loader):
            net.optimizer.zero_grad()
            batch += 1
            data1 = [item.to(device) for item in data]
            y = data1[-1]
            z = net(data1)
            cost = net.loss_function(z, y)

            cost.backward()
            net.optimizer.step()
            epoch_cost += cost.data
        print('\tLoss:\t%.3f' % epoch_cost)
        net.scheduler.step()

        # print(f'epoch:{epoch_num}, cost:{epoch_cost / batch}')
        # <-----------------------------------------------------测试部分
        print('开始测试')
        '''测试'''
        net.eval()
        hit10, hit20, mrr10, mrr20 = [], [], [], []
        _hit10, _hit20, _mrr10, _mrr20 = [], [], [], []

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                data1 = [item.to(device) for item in data]
                y = data1[-1]
                testScore = net(data1)
                sub_scores10 = testScore.topk(10)[1]
                sub_scores20 = testScore.topk(20)[1]
                for score, target in zip(sub_scores10.detach().cpu().numpy(), y.detach().cpu().numpy()):
                    hit10.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr10.append(0)
                    else:
                        mrr10.append(1 / (np.where(score == target)[0][0] + 1))
                for score, target in zip(sub_scores20.detach().cpu().numpy(), y.detach().cpu().numpy()):
                    hit20.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr20.append(0)
                    else:
                        mrr20.append(1 / (np.where(score == target)[0][0] + 1))
            #     sub_scores40 = testScore.topk(40)[1]
            #     x_ = data1[0]
            #     for score, target in zip(sub_scores40.detach().cpu().numpy(), y.detach().cpu().numpy()):

        print('\n')
        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100
        if hit10 > best_result[0]:
            best_result[0] = hit10
        if mrr10 > best_result[2]:
            best_result[2] = mrr10
        if hit20 > best_result[1]:
            best_result[1] = hit20
        if mrr20 > best_result[3]:
            best_result[3] = mrr20
        print('hit10=', hit10)
        print('mrr10=,', mrr10)
        print('hit20=', hit20)
        print('mrr20=,', mrr20)
        print('best result=', best_result)
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)
            if epoch % 100 == 0:
                print('第' + str(epoch) + '次迭代后的epoch_cost=' + str(epoch_cost))
