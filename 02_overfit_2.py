import torch
import numpy as np
import sys
import utils as d2l
print(torch.__version__)

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05    # [200, 1], 0.05

features = torch.randn((n_train + n_test, num_inputs))     # [120, 200]
labels = torch.matmul(features, true_w) + true_b           # [120, 1]
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)       # [120, 1]
train_features, test_features = features[:n_train, :], features[n_train:, :]      # [20, 200], [100, 200]
train_labels, test_labels = labels[:n_train], labels[n_train:]      # [20, 1], [100, 1]


# 定义参数初始化函数，初始化模型参数并且附上梯度
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return (w**2).sum() / 2


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):       # 100
        for X, y in train_iter:    # [1, 200], [1, 1]
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

# 观察过拟合
fit_and_plot(lambd=0)

# 使用权重衰减
fit_and_plot(lambd=3)

