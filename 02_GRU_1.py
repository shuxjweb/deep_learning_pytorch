import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)  # 正态分布
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数    [1027, 256], [256, 256], [256,]
    W_xr, W_hr, b_r = _three()  # 重置门参数    [1027, 256], [256, 256], [256,]
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数   [1027, 256], [256, 256], [256,]

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))      # [256, 1027]
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):  # 隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def gru(inputs, state, params):   # [35, 32, 1027], [1, 32, 256]
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state     # [32, 256]
    outputs = []
    for X in inputs:   # [32,1027]
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)    # [32,256] <- [32,1027]*[1027,256]+[32,256]*[256,256]+[256,]
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)    # [32,256] <- [32,1027]*[1027,256]+[32,256]*[256,256]+[256,]
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)   # [32, 256] <- [32,1027]*[1027,256] + [32,256] * ([32,256].*[256,256]) + [256,]
        H = Z * H + (1 - Z) * H_tilda      # [32, 256]
        Y = torch.matmul(H, W_hq) + b_q    # [32, 1027] <- [32, 256] * [256, 1027] + [1027,]
        outputs.append(Y)
    return outputs, (H,)    # [35, 32, 1027], [32, 256]


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)







