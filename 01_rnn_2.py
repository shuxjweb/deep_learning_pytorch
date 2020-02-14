import torch
import torch.nn as nn
import time
import math
import sys
import utils as d2l


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class) -> [2, 1027]
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result


# 裁剪梯度
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:     # theta=0.01
        for param in params:
            param.grad.data *= (theta / norm)

x = torch.tensor([0, 2])
x_one_hot = one_hot(x, vocab_size)      # [2, 1027]
print(x_one_hot)
print(x_one_hot.shape)
print(x_one_hot.sum(dim=1))

def to_onehot(X, n_class):      # [32, 35], 1027
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)             # [2, 5]
inputs = to_onehot(X, vocab_size)           # [5, 2, 1027]
print(len(inputs), inputs[0].shape)

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数
# num_outputs: q

rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps, batch_size = 35, 2
X = torch.rand(num_steps, batch_size, vocab_size)     # [35, 2, 1027]
state = None
Y, state_new = rnn_layer(X, state)       # [35, 2, 256],  [1, 2, 256]
print(Y.shape, state_new.shape)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)   [1, 1, 1027]
        hiddens, state = self.rnn(X, state)    # [1, 1, 256], [1, 1, 256]
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)  [1, 256]
        output = self.dense(hiddens)      # [1, 1027]
        return output, state

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数  [1, 1027], [1, 1, 256]
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])

model = RNNModel(rnn_layer, vocab_size).to(device)
print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))
# 分开爷爷爷爷爷爷爷爷爷爷

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):     # 250
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state[0].detach_()
                    state[1].detach_()
                else:
                    state.detach_()
            (output, state) = model(X, state)  # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.t())
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)

# epoch 250, perplexity 1.019995, time 0.25 sec
#  - 分开 你爱起 它爱还有开始单像 有何有没有 有什么 有话我回到你爸爸  你想要 你你叫我妈妈妈 这样对吗
#  - 不分开始了我面 家  你一定实现慢休的笑 没古过了我 离开能我满天 手牵手 一步两步三步四步望著天 看星星



