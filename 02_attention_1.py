import math
import torch
import torch.nn as nn

import os


def file_name_walk(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #         print("root", root)  # 当前目录路径
        print("dirs", dirs)  # 当前路径下所有子目录
        print("files", files)  # 当前路径下所有非目录子文件


file_name_walk("fraeng6506")


# Softmax屏蔽
def SequenceMask(X, X_len, value=-1e6):
    maxlen = X.size(1)
    # print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
    mask = torch.arange((maxlen), dtype=torch.float)[None, :] >= X_len[:, None]   # [4, 4]
    # print(mask)
    X[mask] = value
    return X


def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
            except:
                valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)   # [4, 4]

        return softmax(X).reshape(shape)


masked_softmax(torch.rand((2, 2, 4), dtype=torch.float), torch.FloatTensor([2, 3]))

# 超出2维矩阵的乘法
torch.bmm(torch.ones((2, 1, 3), dtype=torch.float), torch.ones((2, 3, 2), dtype=torch.float))


# 点积注意力
# Save to the d2l package.
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):   # [2, 1, 2], [2, 10, 2], [2, 10, 4]
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)   # [2, 1, 10]
        attention_weights = self.dropout(masked_softmax(scores, valid_length))      # [2, 1, 10]
        print("attention_weight\n", attention_weights)
        return torch.bmm(attention_weights, value)


# 测试
atten = DotProductAttention(dropout=0)

keys = torch.ones((2, 10, 2), dtype=torch.float)
values = torch.arange((40), dtype=torch.float).view(1, 10, 4).repeat(2, 1, 1)
atten(torch.ones((2, 1, 2), dtype=torch.float), keys, values, torch.FloatTensor([2, 6]))


# 多层感知机注意力
# Save to the d2l package.
class MLPAttention(nn.Module):
    def __init__(self, units, ipt_dim, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Linear(ipt_dim, units, bias=False)    # [2, 8]
        self.W_q = nn.Linear(ipt_dim, units, bias=False)    # [2, 8]
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):  # [2, 1, 2], [2, 10, 2], [2, 10, 4]
        query, key = self.W_k(query), self.W_q(key)  # [2, 1, 8], [2, 10, 8]
        # print("size",query.size(),key.size())
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.unsqueeze(2) + key.unsqueeze(1)   # [2, 1, 10, 8]
        # print("features:",features.size())  #--------------开启
        scores = self.v(features).squeeze(-1)   # [2, 1, 10]
        attention_weights = self.dropout(masked_softmax(scores, valid_length))      # [2, 1, 10]
        return torch.bmm(attention_weights, value)


# 测试
atten = MLPAttention(ipt_dim=2, units=8, dropout=0)
atten(torch.ones((2, 1, 2), dtype=torch.float), keys, values, torch.FloatTensor([2, 6]))
