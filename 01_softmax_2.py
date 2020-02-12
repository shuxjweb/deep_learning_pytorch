import torch
import torchvision
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import utils as d2l

print(torch.__version__)
print(torchvision.__version__)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='input/FashionMNIST2065')


num_inputs = 784
print(28*28)
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)      # [784, 10]
b = torch.zeros(num_outputs, dtype=torch.float)     # [10,]

num_inputs = 784
print(28*28)
num_outputs = 10

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征

def softmax(X):       # [256, 10]
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))

def net(X):       # [256, 1, 28, 28]
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])     # [2, 3]
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))

def cross_entropy(y_hat, y):     # [256, 10], [256,]
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))     # [256,1]

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

print(accuracy(y_hat, y))

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:    # [256, 1, 28, 28], [256,]
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

print(evaluate_accuracy(test_iter, net))       # 0.1065

num_epochs, lr = 5, 0.1


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):       # 5
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)     # [256, 10]
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


# epoch 1, loss 0.7862, train acc 0.746, test acc 0.792
# epoch 2, loss 0.5719, train acc 0.812, test acc 0.813
# epoch 3, loss 0.5248, train acc 0.826, test acc 0.821
# epoch 4, loss 0.5019, train acc 0.832, test acc 0.825
# epoch 5, loss 0.4849, train acc 0.837, test acc 0.824






