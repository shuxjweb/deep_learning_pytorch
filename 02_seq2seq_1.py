import math
import torch
import torch.nn as nn
import d2l
import torch


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



class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = MLPAttention(num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        #         print("first:",outputs.size(),hidden_state[0].size(),hidden_state[1].size())
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.permute(1, 0, -1), hidden_state, enc_valid_len)
        # outputs.swapaxes(0, 1)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        # ("X.size",X.size())
        X = self.embedding(X).transpose(0, 1)
        #         print("Xembeding.size2",X.size())
        outputs = []
        for l, x in enumerate(X):
            #             print(f"\n{l}-th token")
            #             print("x.first.size()",x.size())
            # query shape: (batch_size, 1, hidden_size)
            # select hidden state of the last rnn layer as query
            query = hidden_state[0][-1].unsqueeze(1)  # np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
            #             print("query enc_outputs, enc_outputs:\n",query.size(), enc_outputs.size(), enc_outputs.size())
            context = self.attention_cell(query, enc_outputs, enc_outputs, enc_valid_len)
            # Concatenate on the feature dimension
            #             print("context.size:",context.size())
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
            #             print("rnn",x.size(), len(hidden_state))
            out, hidden_state = self.rnn(x.transpose(0, 1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state, enc_valid_len]

encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8,
                            num_hiddens=16, num_layers=2)
# encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8,
                                  num_hiddens=16, num_layers=2)
X = torch.zeros((4, 7),dtype=torch.long)
print("batch size=4\nseq_length=7\nhidden dim=16\nnum_layers=2\n")
print('encoder output size:', encoder(X)[0].size())
print('encoder hidden size:', encoder(X)[1][0].size())
print('encoder memory size:', encoder(X)[1][1].size())
state = decoder.init_state(encoder(X), None)
out, state = decoder(X, state)
out.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape






