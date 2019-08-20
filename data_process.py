import random
import torch
import numpy as np
import torch.utils.data as data
from itertools import chain
from corpus import Corpus


def make_vector(texts, text_size, sent_len):  # Pad the conv with 0s to fixed size
    text_vec = []
    for one_text in texts:
        t = []
        for sent in one_text:
            pad_len = max(0, sent_len - len(sent))
            t.append(sent + [0] * pad_len)
        pad_size = max(0, text_size - len(t))
        t.extend([[0] * sent_len] * pad_size)
        text_vec.append(t)
    return torch.LongTensor(text_vec)


class RatingPerUser:
    def __init__(self):
        self.positives = []
        self.negatives = []


class MyDataset(data.Dataset):
    def __init__(self, raw_corpus, ratings, num_negatives=100, sampling=True):
        self.uc_pairs = []
        self.data_label = []
        if not sampling:
            # For small datasets, include all negative instances when evaluating
            for u in xrange(raw_corpus.userNum):
                for c in ratings[u].negatives:
                    self.uc_pairs.append((u, c))
                    self.data_label.append(0)
        for u in xrange(raw_corpus.userNum):
            for c in ratings[u].positives:
                self.uc_pairs.append((u, c))
                self.data_label.append(1)
                if sampling:
                    # For training, developing or for large datasets, each positive instance samples negative instances
                    num = len(ratings[u].negatives)
                    if num == 0:
                        continue
                    for t in xrange(num_negatives):
                        j = np.random.randint(num)
                        self.uc_pairs.append((u, ratings[u].negatives[j]))
                        self.data_label.append(0)
        self.uc_pairs = torch.LongTensor(self.uc_pairs)
        self.data_label = torch.Tensor(self.data_label)

    def __getitem__(self, idx):
        return self.uc_pairs[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_label)


def form_dataset(raw_corpus, batch_size, test_fraction=0.1, sampling=False):
    conv_ids = sorted(raw_corpus.convs.keys(), reverse=True)
    have_ratings = {}
    train_ratings = {}
    test_ratings = {}
    dev_ratings = {}
    for uid in xrange(raw_corpus.userNum):
        have_ratings[uid] = set()
        train_ratings[uid] = RatingPerUser()
        test_ratings[uid] = RatingPerUser()
        dev_ratings[uid] = RatingPerUser()
    count = 0
    for cid in conv_ids:
        for turn in raw_corpus.convs[cid]:
            uid = turn[0]
            have_ratings[uid].add(cid)
            train_ratings[uid].positives.append(cid)
        for uid in raw_corpus.pred_reply[cid]:
            have_ratings[uid].add(cid)
            if count % 2 == 0:  # Half for test, half for dev
                test_ratings[uid].positives.append(cid)
            else:
                dev_ratings[uid].positives.append(cid)
            count += 1

    for uid in xrange(raw_corpus.userNum):
        for cid in xrange(raw_corpus.convNum):
            if cid not in have_ratings[uid]:
                r = random.random()
                if r < test_fraction:
                    test_ratings[uid].negatives.append(cid)
                elif r < 2 * test_fraction:
                    dev_ratings[uid].negatives.append(cid)
                else:
                    train_ratings[uid].negatives.append(cid)

    test_data = MyDataset(raw_corpus, test_ratings, sampling=sampling)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
    dev_data = MyDataset(raw_corpus, dev_ratings)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size, num_workers=0)
    return train_ratings, test_loader, dev_loader


def create_embedding_matrix(filename, corp, embedding_dim=200):
    pretrain_file = 'glove.twitter.27B.200d.txt' if filename[0] == 't' else 'glove.6B.200d.txt'
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    word_idx = corp.r_wordIDs
    vocab_num = corp.wordNum
    weights_matrix = np.zeros((vocab_num, embedding_dim))
    for idx in word_idx.keys():
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)


def create_conv_data(convs):
    conv_turn_size = max([len(convs[c]) for c in convs.keys()])
    conv_sent_len = max([len(sent) for sent in chain.from_iterable([convs[c] for c in convs.keys()])])
    text_vec = []
    for key in sorted(convs.keys()):
        t = []
        for sent in convs[key]:
            pad_len = max(0, conv_sent_len - len(sent))
            t.append(sent + [0] * pad_len)
        pad_size = max(0, conv_turn_size - len(t))
        t.extend([[-1] * 4 + [0] * (conv_sent_len - 4)] * pad_size)
        text_vec.append(t)
    return torch.LongTensor(text_vec)


def create_arc_info(convs, no_time=False):
    conv_num = convs.size(0)
    turn_num = convs.size(1)
    arcs = torch.zeros((2, conv_num, turn_num, turn_num))  # arcs[0]: in arcs, arcs[1]: out arcs
    for i in range(conv_num):
        for turn in convs[i]:
            if turn[1] == -1:
                break
            if turn[1] == 0:
                continue
            if turn[2] >= turn_num:  # For some presenting sequence mistakes
                turn[2] = 0
            # replying relations
            arcs[0, i, turn[1], turn[2]] += 1
            arcs[1, i, turn[2], turn[1]] += 1
            # time series relations
            if not no_time:
                arcs[0, i, turn[1], turn[1]-1] += 1
                arcs[1, i, turn[1]-1, turn[1]] += 1

    return arcs


if __name__ == '__main__':
    corp = Corpus("twitter.data", -1, 0.75)
    test = form_dataset(corp, 10)

