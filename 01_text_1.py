import collections
import re

def read_time_machine():
    with open('input/timemachine7163/timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


lines = read_time_machine()
print('# sentences %d' % len(lines))

def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)        # 分词


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # :
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数

vocab = Vocab(tokens)       # 建立字典
print(list(vocab.token_to_idx.items())[0:10])

for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

# words: ['caressed', 'us', 'rather', 'than', 'submitted', 'to', 'be', 'sat', 'upon', 'and', 'there', 'was', 'that']
# indices: [57, 24, 58, 59, 60, 15, 13, 61, 62, 29, 63, 19, 48]
# words: ['luxurious', 'after', 'dinner', 'atmosphere', 'when', 'thought', 'roams', 'gracefully']
# indices: [64, 65, 66, 67, 68, 69, 70, 71]


######################################################
# import spacy
#
# text = "Mr. Chen doesn't agree with my suggestion."
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(text)
# print([token.text for token in doc])
#
# from nltk.tokenize import word_tokenize
# from nltk import data
# data.path.append('input/nltk_data3784/nltk_data')
# print(word_tokenize(text))

