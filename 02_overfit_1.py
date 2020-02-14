import torch
import numpy as np
import sys
import utils as d2l
print(torch.__version__)

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))      # [200, 1]
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)    # [200, 3]
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)    # [200,]

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    # d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)


num_epochs, loss = 100, torch.nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):   # [100, 3], [100, 3], [100,], [100,]
    # 初始化网络模型
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    # 设置批量大小
    batch_size = min(10, train_labels.shape[0])   # 10
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)  # 设置数据集
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)  # 设置获取数据方式

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 设置优化函数，使用的是随机梯度下降优化
    train_ls, test_ls = [], []
    for _ in range(num_epochs):        # 100
        for X, y in train_iter:  # 取一个批量的数据   [10, 3], [10,]
            l = loss(net(X), y.view(-1, 1))  # 输入到网络中计算输出，并和标签比较求得损失函数
            optimizer.zero_grad()  # 梯度清零，防止梯度累加干扰优化
            l.backward()  # 求梯度
            optimizer.step()  # 迭代优化函数，进行参数优化
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())  # 将训练损失保存到train_ls中
        test_ls.append(loss(net(test_features), test_labels).item())  # 将测试损失保存到test_ls中
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)

# 三阶多项式函数拟合（正常）
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
# final epoch: train loss 8.249050006270409e-05 test loss 0.00010000859037972987
# weight: tensor([[ 1.1973, -3.3995,  5.6001]])
# bias: tensor([4.9990])

# 线性函数拟合（欠拟合）
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
# final epoch: train loss 205.2456817626953 test loss 123.19426727294922
# weight: tensor([[20.4799]])
# bias: tensor([2.1316])

# 训练样本不足（过拟合）
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])
# final epoch: train loss 2.2342777252197266 test loss 200.85494995117188
# weight: tensor([[1.4205, 0.2003, 0.6566]])
# bias: tensor([2.7761])





