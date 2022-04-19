import pickle
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import numpy as np
import math
import time
from torch.autograd import Variable

# np.random.seed(2)
# torch.manual_seed(2)
# torch.cuda.manual_seed_all(2)
# torch.cuda.manual_seed_all(2)
# random.seed(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu' if torch.cuda.is_available() else 'cuda:1')
class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        '''x表示输入，（M,N）N个样本，M代表总类数，每一个类的概率log p
            target 表示label(M,)
        '''
        assert x.size(1) == self.size
        x = x.log()
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            session_len,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(session_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat, last_nodes, mask):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat * mask
        feat = self.feat_drop(feat)
        feat = feat * mask
        feat_u = self.fc_u(feat)
        feat_u = feat_u * mask
        feat_v = self.fc_v(last_nodes)  # (batch_size * embedding_size)
        feat_v = torch.unsqueeze(feat_v, 1)

        e = self.fc_e(self.sigmoid(feat_u + feat_v))

        # attention_mask = mask + 1
        # attention_mask[attention_mask == 2] = 0
        # attention_mask = attention_mask * - (pow(math.e, 32))
        # attention_mask[attention_mask == 0] = 1
        # e = e + (attention_mask - mask)
        mask1 = (mask - 1) * 2e32
        e = e + mask1
        beta = self.softmax(e)
        feat_norm = feat * beta
        feat_norm = feat_norm * mask

        rst = torch.sum(feat_norm, dim=1)

        if self.fc_out is not None:
            rst = self.fc_out(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, session_length):
        super().__init__()
        self.embedding_size = embedding_size
        self.session_len = session_length
        self.wq = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.wk = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.wv = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        self.d = math.sqrt(embedding_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.reset_parameters()

    def forward(self, session, mask):
        att_mask = (mask - 1) * 2e32

        q = self.relu(session @ self.wq)
        k = self.relu(session @ self.wk).transpose(2, 1)
        v = self.relu(session @ self.wv)
        alpha = q @ k / self.d
        alpha = alpha + att_mask + att_mask.transpose(2, 1)
        alpha = self.softmax(alpha) * mask
        out = alpha @ v
        return out

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



class SeqGGNN(Module):
    def __init__(
            self,
            layers=1,
            embedding_size=100,
            feat_drop=0
    ):
        super(SeqGGNN, self).__init__()
        self.conv_layer = nn.Conv2d(1, 1, (1, 2))  # kernel size x 3
        self.layers = layers
        self.embedding_size = embedding_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.W_alpha = nn.Linear(2 * embedding_size, 1, bias=True)
        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.W1 = nn.Sequential(nn.Linear(embedding_size, embedding_size, bias=True),
                                nn.ReLU())
        self.W2 = nn.Sequential(nn.Linear(embedding_size, embedding_size, bias=True),
                                nn.ReLU())
        self.reset_parameters()

    def gnn_cell(self, x, graph_in, graph_out):
        # 小trick diag(x @ neighbor.t()) = torch.unsqueeze(torch.sum((x * neighbor), dim=1),0)

        in_neighbor = torch.sparse.mm(graph_in, x)
        out_neighbor = torch.sparse.mm(graph_out, x)

        in_score = torch.squeeze(torch.sum((self.W1(x * in_neighbor) / math.sqrt(self.embedding_size)), dim=1), 0)
        out_score = torch.squeeze(torch.sum((self.W2(x * out_neighbor) / math.sqrt(self.embedding_size)), dim=1), 0)
        score = self.softmax(torch.stack((in_score, out_score), dim=1))
        score_in = torch.unsqueeze(score[:, 0], dim=-1)
        score_out = torch.unsqueeze(score[:, 1], dim=-1)
        neighbor = in_neighbor * score_in + out_neighbor * score_out

        # # conv
        # agg = torch.stack((x, neighbor), dim=2)
        # agg = torch.unsqueeze(agg, 1)
        # out_conv = self.conv_layer(agg)
        # emb = self.feat_drop(torch.squeeze(out_conv))
        agg = torch.stack((x, neighbor), dim=2)
        agg = torch.unsqueeze(agg, 1)
        # print('co agg.shape', agg.shape)
        out_conv = self.conv_layer(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
        # sum
        # emb = x + neighbor

        # neighbor_feature = torch.sparse.mm(graph_in, x) + torch.sparse.mm(graph_out, x)
        # agg = torch.stack((x, neighbor_feature), dim=2)  # n x dim x 3
        # agg = torch.unsqueeze(agg, 1)
        # out_conv = self.conv_layer(agg)
        # emb = self.feat_drop(torch.squeeze(out_conv))
        return emb

    def forward(self, x, graph_in, graph_out):
        for i in range(self.layers):
            x = self.gnn_cell(x, graph_in, graph_out)
        return x

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class CoGGNN(Module):
    def __init__(
            self,
            layers=1,
            embedding_size=100,
            feat_drop=0
    ):
        super(CoGGNN, self).__init__()
        self.conv_layer = nn.Conv2d(1, 1, (1, 2))  # kernel size x 3
        self.layers = layers
        self.embedding_size = embedding_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.reset_parameters()

    def gnn_cell(self, x, graph_in):
        neighbor_feature = torch.sparse.mm(graph_in, x)
        agg = torch.stack((neighbor_feature, x), dim=2)  # n x dim x 3
        agg = torch.unsqueeze(agg, 1)
        # print('seq agg.shape', agg.shape)
        out_conv = self.conv_layer(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
        # print('seq emb.shape', emb.shape)
        return emb

    def forward(self, x, graph_in):
        for i in range(self.layers):
            x = self.gnn_cell(x, graph_in)
        return x

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



class InGGNN(Module):
    def __init__(
            self,
            layers=2,
            embedding_size=100,
            feat_drop=0
    ):
        super(InGGNN, self).__init__()
        self.layers = layers
        self.embedding_size = embedding_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def gnn_cell(self, x, graph_in, graph_out):
        # in_feature = torch.sparse.mm(graph_in,x)
        # out_feature = torch.sparse.mm(graph_out,x)
        # neighbor_feature = torch.stack((in_feature,out_feature),dim=1)
        # emb = self.feat_drop(torch.mean(neighbor_feature,dim=1)) + x
        #
        # return emb
        neighbor_feature = torch.sparse.mm(graph_in, x) + torch.sparse.mm(graph_out, x)
        agg = torch.stack((x, neighbor_feature), dim=1)
        emb = self.feat_drop(torch.mean(agg, dim=1))


        return emb

    def forward(self, x, graph_in, graph_out):
        for i in range(self.layers):
            x = self.gnn_cell(x, graph_in, graph_out)
        return x

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class MRGSR(Module):
    def __init__(
            self,
            item_num,
            embedding_size,
            seq_in,
            seq_out,
            co_in,
            co_out,
            in_in,
            in_out,
#             graph_in,
            session_len=19,
            gnn_layers=1,
            gnn_drop=0,
            local_drop=0,
            out_channel=1,
            lr=0.001,
            w_decay=1e-4,
            lr_dc_step=3,
            lr_dc=0.1,
            alpha=0,
            device='cpu'
    ):
        super(MRGSR, self).__init__()
        # 图
        self.device = device
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.co_in = co_in
        self.co_out = co_out
        self.in_in = in_in
        self.in_out = in_out
#         self.graph_in = graph_in

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, batch_first=True)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=embedding_size, bias=False, batch_first=True,
                          bidirectional=True)
        self.qkv_attention = SelfAttention(embedding_size=embedding_size, session_length=session_len)
        self.item_num = item_num
        self.gnn_layers = gnn_layers
        self.out_channel = out_channel
        self.embedding_size = embedding_size
        self.session_len = session_len
        self.relu = nn.ReLU()
        self.LeakyRelu = nn.LeakyReLU(negative_slope=alpha)
        self.linearTransform2 = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.mlp2_seq = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.mlp2_co = nn.Linear(2 * embedding_size, embedding_size, bias=False)

        self.mlp2 = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.mlp3 = nn.Linear(3 * embedding_size, embedding_size, bias=False)
        self.mlp4 = nn.Linear(4 * embedding_size, embedding_size, bias=False)
        self.Wsc = nn.Parameter(torch.Tensor(2 * embedding_size, embedding_size))
        self.Wpi = nn.Parameter(torch.Tensor(2 * embedding_size, embedding_size))
        self.Wni = nn.Parameter(torch.Tensor(2 * embedding_size, embedding_size))

        self.mlp_p_ls = nn.Linear(2 * embedding_size, embedding_size)
        self.mlp_n_ls = nn.Linear(2 * embedding_size, embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.read_out = AttnReadout(
            input_dim=embedding_size,
            hidden_dim=embedding_size,
            output_dim=embedding_size,
            session_len=session_len,
            batch_norm=True,
            feat_drop=local_drop,
            activation=nn.PReLU(embedding_size),
        )

        # self.drop_rate = feat_drop
        self.gnn_drop_rate = gnn_drop
        self.feat_drop = nn.Dropout(local_drop)

        self.item_embedding = nn.Embedding(  # item_embedding
            item_num + 1 + 1,
            embedding_size,
            max_norm=1,
            padding_idx=item_num
        )
        self.pos_embedding = nn.Embedding(500, self.embedding_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.embedding_size, self.embedding_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.embedding_size, 1))
        self.glu1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.glu2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.W_seq = nn.Parameter(torch.Tensor(embedding_size, 1))
        self.W_co = nn.Parameter(torch.Tensor(embedding_size, 1))
        self.W_in = nn.Parameter(torch.Tensor(embedding_size, 1))
        self.W_id = nn.Parameter(torch.Tensor(embedding_size, 1))

        self.w_n_1 = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))
        self.w_n_2 = nn.Parameter(torch.Tensor(self.embedding_size, 1))
        self.glun1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.glun2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        # self.loss_function = LabelSmoothingLoss(self.item_num, 0.1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=w_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_dc_step, gamma=lr_dc)
        self.reset_parameters()

    def get_seq_emb(self, emb):
        seqGNN = SeqGGNN(layers=self.gnn_layers,
                         embedding_size=self.embedding_size,
                         feat_drop=self.gnn_drop_rate).to(self.device)
        item_emb = seqGNN(emb, self.seq_in, self.seq_out)
        return item_emb

    def get_co_emb(self, emb):
        coGNN = CoGGNN(layers=self.gnn_layers,
                       embedding_size=self.embedding_size,
                       feat_drop=self.gnn_drop_rate).to(self.device)
        item_emb = coGNN(emb, self.co_in)
        return item_emb

    def get_in_emb(self, emb):
        InGNN = InGGNN(layers=self.gnn_layers,
                       embedding_size=self.embedding_size,
                       feat_drop=self.gnn_drop_rate).to(self.device)
        item_emb = InGNN(emb, self.in_in, self.in_out)
        return item_emb

    def SSL_task(self, interest, user):
        def row_column_shuffle(embedding):
            row_index = [i for i in range(embedding.shape[0])]
            random.shuffle(row_index)
            corrupted_embedding = embedding[row_index].T
            col_index = [i for i in range(corrupted_embedding.shape[0])]
            random.shuffle(col_index)
            corrupted_embedding = corrupted_embedding[col_index]
            return corrupted_embedding.T

        def score(x1, x2):
            return (x1 * x2).mean(1)

        interest = torch.nn.functional.normalize(interest, p=2, dim=1)
        user = torch.nn.functional.normalize(user, p=2, dim=1)

        pos = score(interest, user)
        neg1 = score(interest, row_column_shuffle(user))
        neg2 = score(user, row_column_shuffle(interest))
        # loss = -torch.log(torch.sigmoid(pos)) - torch.log(1 - torch.sigmoid(neg1)) - torch.log(1 - torch.sigmoid(neg2))
        con_loss = torch.mean(
            -torch.log(torch.sigmoid(pos)) - torch.log(1 - torch.sigmoid(neg1)) - torch.log(1 - torch.sigmoid(neg2)))
        return con_loss

    def get_session_representation(self, hidden, mask):
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = torch.flip(pos_emb, [0])
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        return select

    def get_neg_session_representation(self, hidden, mask):
        len = hidden.shape[1]
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(hidden, self.w_n_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glun1(nh) + self.glun2(hs))
        beta = torch.matmul(nh, self.w_n_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        return select

    def get_session_representation_with_position(self, target, hidden, mask):
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[1:len+1]
        pos_emb = torch.flip(pos_emb, [0])
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        target_pos = self.pos_embedding.weight[0]
        # hs = id_emb_[:, -1, :]
        hs = target
        hs = torch.matmul(torch.cat([target_pos, hs], -1), self.w_1)
        hs = torch.tanh(hs)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        return select

    def get_session_representation_without_position(self, id_emb_, hidden, mask):
        short = id_emb_[:, -1, :]
        relation_short = hidden[:, -1, :]
        negative_long_term = self.read_out(hidden, relation_short, mask)
        negative_fuse_long_short = torch.cat((negative_long_term, short), dim=-1)
        negative_session_representation = self.mlp_n_ls(self.feat_drop(negative_fuse_long_short))
        return negative_session_representation

    def forward(self, data):
        items, mask, _ = data  # items == seq
        mask = torch.unsqueeze(mask, -1)
        item_embedding = self.item_embedding.weight[:-1]
        seq_emb = self.get_seq_emb(item_embedding)
        co_emb = self.get_co_emb(item_embedding)
        in_emb = self.get_in_emb(item_embedding)

        id_emb = item_embedding[items] * mask
        item_seq = seq_emb[items] * mask
        item_co = co_emb[items] * mask
        item_in = in_emb[items] * mask
        target_item = self.item_embedding.weight[-1]
        # print(target_item.shape)
        # print(id_emb.shape)
        # assert 1==2
        # seq = self.sigmoid(item_seq @ self.W_seq + id_emb @ self.W_id)
        # seq_item = seq * item_seq + (1 - seq) * id_emb
        # co = self.sigmoid(item_co @ self.W_co + id_emb @ self.W_id)
        # co_item = co * item_co + (1 - co) * id_emb
        # in_s = self.sigmoid(item_in @ self.W_in + id_emb @ self.W_id)
        # in_item = in_s * item_in + (1 - in_s) * id_emb
        #
        # seq_session = self.get_session_representation_with_position(target_item,seq_item,mask)
        # co_session = self.get_session_representation_without_position(id_emb,co_item,mask)
        # in_session = self.get_session_representation_without_position(id_emb,in_item,mask)
        #
        # session = seq_session + co_session - self.LeakyRelu(in_session)
        # score = session @ self.item_embedding.weight[:-2].t()
        # return score

        # gated
        seq_co_score = self.sigmoid((item_seq+item_co) @ self.W_seq + id_emb @ self.W_id)
        positive_emb = seq_co_score * (item_seq + item_co) + (1 - seq_co_score) * id_emb
        neg_score = self.sigmoid(item_in @ self.W_in + id_emb @ self.W_id)
        negative_emb = neg_score * item_in + (1 - neg_score) * id_emb

        positive_session_representation = self.get_session_representation_with_position(target_item, positive_emb, mask)

        negative_session_representation = self.get_session_representation_without_position(id_emb, negative_emb, mask)
        positive_score = positive_session_representation @ self.item_embedding.weight[:-2].t()
        negative_score = negative_session_representation @ self.item_embedding.weight[:-2].t()

        score = positive_score - self.LeakyRelu(negative_score)
        return score



    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)