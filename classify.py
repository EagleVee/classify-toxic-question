#!/usr/bin/env pythonr
# coding: utf-8
import os
import random
import re
import time
from collections import Counter
from itertools import chain
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import shuffle
from torch import optim
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
from model import ToxicModel
from util import AverageMeter, count_parameters, margin_score, report_perf, get_gpu_memory_usage, avg

# constants
embedding_glove = './input/embeddings/glove.840B.300d/glove.840B.300d.txt'
embedding_fasttext = './input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
embedding_para = './input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
embedding_w2v = './input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
train_path = './input/train.csv'
test_path = './input/test.csv'

mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would",
                "i'd've": "i would have", "i'll": "i will", "i'll've": "I will have", "i'm": "i am",
                "i've": "I have", "isn't": "is not", "it'd": "it would",
                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is",
                "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                "they'll've": "they will have", "they're": "they are", "they've": "they have",
                "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color',
                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                'youtu ': 'youtube ', 'qoura': 'quora', 'sallary': 'salary', 'whta': 'what',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doi': 'do I',
                'thebest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                'etherium': 'ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

puncts = '\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
for p in puncts:
    punct_mapping[p] = ' %s ' % p

p = re.compile('(\[ math \]).+(\[ / math \])')
p_space = re.compile(r'[^\x20-\x7e]')


# seeding
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    random.seed(seed + 4)


# data loading & pre-processing

def clean_text(text):
    # clean latex maths
    text = p.sub(' [ math ] ', text)
    # clean invisible chars
    text = p_space.sub(r'', text)
    # clean punctuations
    for punct in punct_mapping:
        if punct in text:
            text = text.replace(punct, punct_mapping[punct])
    tokens = []
    for token in text.split():
        # replace contractions & correct misspells
        token = mispell_dict.get(token.lower(), token)
        tokens.append(token)
    text = ' '.join(tokens)
    return text


def load_data(train_path=train_path, test_path=test_path, debug=False):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if debug:
        train_df = train_df[:10000]
        test_df = test_df[:10000]
    s = time.time()
    train_df['question_text'] = train_df['question_text'].apply(clean_text)
    test_df['question_text'] = test_df['question_text'].apply(clean_text)
    print('preprocssing {}s'.format(time.time() - s))
    return train_df, test_df


# vocabulary functions
def build_counter(sents, splited=False):
    counter = Counter()
    for sent in tqdm(sents, ascii=True, desc='building conuter'):
        if splited:
            counter.update(sent)
        else:
            counter.update(sent.split())
    return counter


def build_vocab(counter, max_vocab_size):
    vocab = {'token2id': {'<PAD>': 0, '<UNK>': max_vocab_size + 1}}
    vocab['token2id'].update(
        {token: _id + 1 for _id, (token, count) in
         tqdm(enumerate(counter.most_common(max_vocab_size)), desc='building vocab')})
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    return vocab


def tokens2ids(tokens, token2id):
    seq = []
    for token in tokens:
        token_id = token2id.get(token, len(token2id) - 1)
        seq.append(token_id)
    return seq


#  data set
class TextDataset(Dataset):
    def __init__(self, df, vocab=None, num_max=None, max_seq_len=100,
                 max_vocab_size=95000):
        if num_max is not None:
            df = df[:num_max]

        self.src_sents = df['question_text'].tolist()
        self.qids = df['qid'].values
        if vocab is None:
            src_counter = build_counter(self.src_sents)
            vocab = build_vocab(src_counter, max_vocab_size)
        self.vocab = vocab
        if 'src_seqs' not in df.columns:
            self.src_seqs = []
            for sent in tqdm(self.src_sents, desc='tokenize'):
                seq = tokens2ids(sent.split()[:max_seq_len], vocab['token2id'])
                self.src_seqs.append(seq)
        else:
            self.src_seqs = df['src_seqs'].tolist()
        if 'target' in df.columns:
            self.targets = df['target'].values
        else:
            self.targets = np.random.randint(2, size=(len(self.src_sents),))
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.src_sents)

    # for bucket iterator
    def get_keys(self):
        lens = np.fromiter(
            tqdm(((min(self.max_seq_len, len(c.split()))) for c in self.src_sents), desc='generate lens'),
            dtype=np.int32)
        return lens

    def __getitem__(self, index):
        return self.qids[index], self.src_sents[index], self.src_seqs[index], self.targets[index]


#  dynamic padding
def _pad_sequences(seqs):
    lens = [len(seq) for seq in seqs]
    max_len = max(lens)

    padded_seqs = torch.zeros(len(seqs), max_len).long()
    for i, seq in enumerate(seqs):
        end = lens[i]
        padded_seqs[i, :end] = torch.LongTensor(seq)
    return padded_seqs, lens


def collate_fn(data):
    qids, src_sents, src_seqs, targets, = zip(*data)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    return qids, src_sents, src_seqs, src_lens, torch.FloatTensor(targets)


#  bucket iterator
def divide_chunks(l, n):
    if n == len(l):
        yield np.arange(len(l), dtype=np.int32), l
    else:
        # looping till length l
        for i in range(0, len(l), n):
            data = l[i:i + n]
            yield np.arange(i, i + len(data), dtype=np.int32), data


def prepare_buckets(lens, bucket_size, batch_size, shuffle_data=True, indices=None):
    lens = -lens
    assert bucket_size % batch_size == 0 or bucket_size == len(lens)
    if indices is None:
        if shuffle_data:
            indices = shuffle(np.arange(len(lens), dtype=np.int32))
            lens = lens[indices]
        else:
            indices = np.arange(len(lens), dtype=np.int32)
    new_indices = []
    extra_batch = None
    for chunk_index, chunk in (divide_chunks(lens, bucket_size)):
        # sort indices in bucket by descending order of length
        indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
        batches = []
        for _, batch in divide_chunks(indices_sorted, batch_size):
            if len(batch) == batch_size:
                batches.append(batch.tolist())
            else:
                assert extra_batch is None
                assert batch is not None
                extra_batch = batch
        # shuffling batches within buckets
        if shuffle_data:
            batches = shuffle(batches)
        for batch in batches:
            new_indices.extend(batch)

    if extra_batch is not None:
        new_indices.extend(extra_batch)
    return indices[new_indices]


class BucketSampler(Sampler):

    def __init__(self, data_source, sort_keys, bucket_size=None, batch_size=1536, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        if not shuffle_data:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=self.shuffle)
        else:
            self.index = None
        self.weights = None

    def set_weights(self, w):
        assert w >= 0
        total = np.sum(w)
        if total != 1:
            w = w / total
        self.weights = w

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)

            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=self.shuffle, indices=indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)


# embedding stuffs
def read_embedding(embedding_file):
    """
    read embedding file into a dictionary
    each line of the embedding file should in the format like  word 0.13 0.22 ... 0.44
    :param embedding_file: path of the embedding.
    :return: a dictionary of word to its embedding (numpy array)
    """
    if os.path.basename(embedding_file) != 'wiki-news-300d-1M.vec':
        skip_head = None
    else:
        skip_head = 0
    if os.path.basename(embedding_file) == 'paragram_300_sl999.txt':
        encoding = 'latin'
    else:
        encoding = 'utf-8'
    embeddings_index = {}
    t_chunks = pd.read_csv(embedding_file, index_col=0, skiprows=skip_head, encoding=encoding, sep=' ', header=None,
                           quoting=3,
                           doublequote=False, quotechar=None, engine='c', na_filter=False, low_memory=True,
                           chunksize=10000)
    for t in t_chunks:
        for k, v in zip(t.index.values, t.values):
            embeddings_index[k] = v.astype(np.float32)
    return embeddings_index


def get_emb(embedding_index, word, word_raw):
    if word == word_raw:
        return None
    else:
        return embedding_index.get(word, None)


def embedding2numpy(embedding_path, word_index, num_words, embed_size, emb_mean=0., emb_std=0.5,
                    report_stats=False):
    embedding_index = read_embedding(embedding_path)
    num_words = min(num_words + 2, len(word_index))
    if report_stats:
        all_coefs = []
        for v in embedding_index.values():
            all_coefs.append(v.reshape([-1, 1]))
        all_coefs = np.concatenate(all_coefs)
        print(all_coefs.mean(), all_coefs.std(), np.linalg.norm(all_coefs, axis=-1).mean())
    embedding_matrix = np.zeros((num_words, embed_size), dtype=np.float32)
    oov = 0
    oov_cap = 0
    oov_upper = 0
    oov_lower = 0
    for word, i in word_index.items():
        if i == 0:  # padding
            continue
        if i >= num_words:
            continue
        embedding_vector = embedding_index.get(word, None)
        if embedding_vector is None:
            embedding_vector = get_emb(embedding_index, word.lower(), word)
            if embedding_vector is None:
                embedding_vector = get_emb(embedding_index, word.upper(), word)
                if embedding_vector is None:
                    embedding_vector = get_emb(embedding_index, word.capitalize(), word)
                    if embedding_vector is None:
                        oov += 1
                        # embedding_vector = (np.zeros((1, embed_size)))
                        embedding_vector = np.random.normal(emb_mean, emb_std, size=(1, embed_size))
                    else:
                        oov_lower += 1
                else:
                    oov_upper += 1
            else:
                oov_cap += 1

        embedding_matrix[i] = embedding_vector

    print('oov %d/%d/%d/%d/%d' % (oov, oov_cap, oov_upper, oov_lower, len(word_index)))
    return embedding_matrix


def load_embedding(vocab, max_vocab_size, embed_size):
    # load embedding
    embedding_matrix1 = embedding2numpy(embedding_glove, vocab['token2id'], max_vocab_size, embed_size,
                                        emb_mean=-0.005838499, emb_std=0.48782197, report_stats=False)
    # -0.005838499 0.48782197 0.37823704
    # oov 9196
    # embedding_matrix2 = embedding2numpy(embedding_fasttext, vocab.token2id, max_vocab_size, embed_size,
    #                                    report_stats=False, emb_mean=-0.0033469985, emb_std=0.109855495, )
    # -0.0033469985 0.109855495 0.07475414
    # oov 12885
    embedding_matrix2 = embedding2numpy(embedding_para, vocab['token2id'], max_vocab_size, embed_size,
                                        emb_mean=-0.0053247833, emb_std=0.49346462, report_stats=False)
    # -0.0053247833 0.49346462 0.3828983
    # oov 9061
    # embedding_w2v
    # -0.003527845 0.13315111 0.09407869
    # oov 18927
    return [embedding_matrix1, embedding_matrix2]


# cyclic learning rate
def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


class CyclicLR:
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.002, step_size=300., mode='triangular',
                 gamma=0.99994, scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self):
        if self.clr_iterations == 0:
            set_lr(self.optimizer, self.base_lr)
        else:
            set_lr(self.optimizer, self.clr())

    def on_batch_end(self):
        self.trn_iterations += 1
        self.clr_iterations += 1
        set_lr(self.optimizer, self.clr())


# evaluation
def eval_model(model, data_iter, device, order_index=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in data_iter:
            qid_batch, src_sents, src_seqs, src_lens, tgts = batch_data
            src_seqs = src_seqs.to(device)
            out = model(src_seqs, src_lens, return_logits=False)
            predictions.append(out)
    predictions = torch.cat(predictions, dim=0)
    if order_index is not None:
        predictions = predictions[order_index]
    predictions = predictions.to('cpu').numpy().ravel()
    return predictions


# cross validation
def cv(train_df, test_df, device=None, n_folds=10, shared_resources=None, share=True, **kwargs):
    if device is None:
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    max_vocab_size = kwargs['max_vocab_size']
    embed_size = kwargs['embed_size']
    threshold = kwargs['threshold']
    max_seq_len = kwargs['max_seq_len']
    if shared_resources is None:
        shared_resources = {}
    if share:
        if 'vocab' not in shared_resources:
            # also include the test set

            counter = build_counter(chain(train_df['question_text'], test_df['question_text']))
            vocab = build_vocab(counter, max_vocab_size=max_vocab_size)
            shared_resources['vocab'] = vocab
            # tokenize sentences
            seqs = []
            for sent in tqdm(train_df['question_text'], desc='tokenize'):
                seq = tokens2ids(sent.split()[:max_seq_len], vocab['token2id'])
                seqs.append(seq)
            train_df['src_seqs'] = seqs
            seqs = []
            for sent in tqdm(test_df['question_text'], desc='tokenize'):
                seq = tokens2ids(sent.split()[:max_seq_len], vocab['token2id'])
                seqs.append(seq)
            test_df['src_seqs'] = seqs
    if 'embedding_matrix' not in shared_resources:
        embedding_matrix = load_embedding(shared_resources['vocab'], max_vocab_size, embed_size)
        shared_resources['embedding_matrix'] = embedding_matrix
    splits = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True).split(train_df['target'], train_df['target']))
    scores = []
    best_threshold = []
    best_threshold_global = None
    best_score = -1
    predictions_train_reduced = []
    targets_train = []
    predictions_tes_reduced = np.zeros((len(test_df), n_folds))
    predictions_te = np.zeros((len(test_df),))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        grow_df = train_df.iloc[train_idx].reset_index(drop=True)
        dev_df = train_df.iloc[valid_idx].reset_index(drop=True)
        predictions_te_i, predictions_va, targets_va, best_threshold_i = main(grow_df, dev_df, test_df, device,
                                                                              **kwargs,
                                                                              idx=idx,
                                                                              shared_resources=shared_resources,
                                                                              return_reduced=True)
        # predictions_va_raw shape (#len_va,n_models)
        predictions_tes_reduced[:, idx] = predictions_te_i
        scores.append([f1_score(targets_va, predictions_va > threshold), roc_auc_score(targets_va, predictions_va)])
        best_threshold.append(best_threshold_i)
        predictions_te += predictions_te_i / n_folds
        predictions_train_reduced.append(predictions_va)
        targets_train.append(targets_va)
    # calculate model coefficient
    coeff = (np.corrcoef(predictions_tes_reduced, rowvar=False).sum() - n_folds) / n_folds / (n_folds - 1)
    # create data set for stacking
    predictions_train_reduced = np.concatenate(predictions_train_reduced)
    targets_train = np.concatenate(targets_train)  # len_train
    # train optimal combining weights

    # simple average
    for t in np.arange(0, 1, 0.01):
        score = f1_score(targets_train, predictions_train_reduced > t)
        if score > best_score:
            best_score = score
            best_threshold_global = t
    print('avg of best threshold {} macro-f1 best threshold {} best score {}'.format(best_threshold,
                                                                                     best_threshold_global, best_score))
    return predictions_te, predictions_te, scores, best_threshold_global, coeff


# main routine
model_path = './model.pth'


def main(train_df, valid_df, test_df, device=None, epochs=3, fine_tuning_epochs=3, batch_size=512, learning_rate=0.001,
         learning_rate_max_offset=0.001, dropout=0.1,
         threshold=None,
         max_vocab_size=95000, embed_size=300, max_seq_len=70, print_every_step=500, idx=0, shared_resources=None,
         return_reduced=True):
    if device is None:
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    if shared_resources is None:
        shared_resources = {}
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mean_len = AverageMeter()
    # build vocab of raw df

    if 'vocab' not in shared_resources:
        counter = build_counter(chain(train_df['question_text'], test_df['question_text']))
        vocab = build_vocab(counter, max_vocab_size=max_vocab_size)
    else:
        vocab = shared_resources['vocab']
    if 'embedding_matrix' not in shared_resources:
        embedding_matrix = load_embedding(vocab, max_vocab_size, embed_size)
    else:
        embedding_matrix = shared_resources['embedding_matrix']
    # create test dataset
    test_dataset = TextDataset(test_df, vocab=vocab, max_seq_len=max_seq_len)
    tb = BucketSampler(test_dataset, test_dataset.get_keys(), batch_size=batch_size,
                       shuffle_data=False)
    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           sampler=tb,
                           # shuffle=False,
                           num_workers=0,
                           collate_fn=collate_fn)

    train_dataset = TextDataset(train_df, vocab=vocab, max_seq_len=max_seq_len)
    # keys = train_dataset.get_keys()  # for bucket sorting
    valid_dataset = TextDataset(valid_df, vocab=vocab, max_seq_len=max_seq_len)
    vb = BucketSampler(valid_dataset, valid_dataset.get_keys(), batch_size=batch_size,
                       shuffle_data=False)
    valid_index_reverse = vb.get_reverse_indexes()
    # init model and optimizers

    model = ToxicModel(device, hidden_dim=256, hidden_dim_fc=16, dropout=dropout,
                       embedding_matrixs=embedding_matrix,
                       vocab_size=len(vocab['token2id']),
                       embedding_dim=embed_size, max_seq_len=max_seq_len)
    if os.path.exists(model_path):
        print('loading model...')
        model.load_state_dict(torch.load(model_path))  # get model if already in
        print('model loaded')
    if idx == 0:
        print(model)
        print('total trainable {}'.format(count_parameters(model)))
    model = model.to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    # init iterator
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            # shuffle=True,
                            # sampler=NegativeSubSampler(train_dataset, train_dataset.targets),
                            sampler=BucketSampler(train_dataset, train_dataset.get_keys(), bucket_size=batch_size * 20,
                                                  batch_size=batch_size),
                            num_workers=0,
                            collate_fn=collate_fn)

    valid_iter = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            sampler=vb,
                            # shuffle=False,
                            collate_fn=collate_fn)

    # train model

    loss_list = []
    global_steps = 0
    total_steps = epochs * len(train_iter)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    end = time.time()
    predictions_tes = []
    predictions_vas = []
    n_fge = 0
    clr = CyclicLR(optimizer, base_lr=learning_rate, max_lr=learning_rate + learning_rate_max_offset,
                   step_size=300, mode='exp_range')
    clr.on_train_begin()
    fine_tuning_epochs = epochs - fine_tuning_epochs
    predictions_te = None
    for epoch in tqdm(range(epochs)):

        fine_tuning = epoch >= fine_tuning_epochs
        start_fine_tuning = fine_tuning_epochs == epoch
        if start_fine_tuning:
            model.enable_learning_embedding()
            optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
            # fine tuning embedding layer
            global_steps = 0
            total_steps = (epochs - fine_tuning_epochs) * len(train_iter)
            clr = CyclicLR(optimizer, base_lr=learning_rate, max_lr=learning_rate + learning_rate_max_offset,
                           step_size=int(len(train_iter) / 8))
            clr.on_train_begin()
            predictions_te = np.zeros((len(test_df),))
            predictions_va = np.zeros((len(valid_dataset.targets),))
        for batch_data in train_iter:
            data_time.update(time.time() - end)
            qids, src_sents, src_seqs, src_lens, tgts = batch_data
            mean_len.update(sum(src_lens))
            src_seqs = src_seqs.to(device)
            tgts = tgts.to(device)
            model.train()
            optimizer.zero_grad()

            out = model(src_seqs, src_lens, return_logits=True).view(-1)
            loss = loss_fn(out, tgts)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().to('cpu').item())

            global_steps += 1
            batch_time.update(time.time() - end)
            end = time.time()
            if global_steps % print_every_step == 0:
                curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
                print('Global step: {}/{} Total loss: {:.4f}  Current GPU memory '
                      'usage: {} maxlen {} '.format(global_steps, total_steps, avg(loss_list), curr_gpu_memory_usage,
                                                    mean_len.avg))
                loss_list = []

            if fine_tuning and global_steps % (2 * clr.step_size) == 0:
                predictions_te_tmp2 = eval_model(model, test_iter, device)
                predictions_va_tmp2 = eval_model(model, valid_iter, device, valid_index_reverse)
                report_perf(valid_dataset, predictions_va_tmp2, threshold, idx, epoch,
                            desc='val set mean')
                predictions_te = predictions_te * n_fge + (
                    predictions_te_tmp2)
                predictions_va = predictions_va * n_fge + (
                    predictions_va_tmp2)
                predictions_te /= n_fge + 1
                predictions_va /= n_fge + 1
                report_perf(valid_dataset, predictions_va, threshold, idx, epoch
                            , desc='val set (fge)')
                predictions_tes.append(predictions_te_tmp2.reshape([-1, 1]))
                predictions_vas.append(predictions_va_tmp2.reshape([-1, 1]))
                n_fge += 1

            clr.on_batch_end()
        if not fine_tuning:
            predictions_va = eval_model(model, valid_iter, device, valid_index_reverse)
            report_perf(valid_dataset, predictions_va, threshold, idx, epoch)
    # reorder index
    torch.save(model.state_dict(), model_path)
    if predictions_te is not None:
        predictions_te = predictions_te[tb.get_reverse_indexes()]
    else:
        predictions_te = eval_model(model, test_iter, device, tb.get_reverse_indexes())
    best_score = -1
    best_threshold = None
    for t in np.arange(0, 1, 0.01):
        score = f1_score(valid_dataset.targets, predictions_va > t)
        if score > best_score:
            best_score = score
            best_threshold = t
    print('best threshold on validation set: {:.2f} score {:.4f}'.format(best_threshold, best_score))
    if not return_reduced and len(predictions_vas) > 0:
        predictions_te = np.concatenate(predictions_tes, axis=1)
        predictions_te = predictions_te[tb.get_reverse_indexes(), :]
        predictions_va = np.concatenate(predictions_vas, axis=1)

    # make predictions
    return predictions_te, predictions_va, valid_dataset.targets, best_threshold


question_path = './question.txt'


if __name__ == '__main__':
    # seeding
    set_seed(233)
    epochs = 8
    batch_size = 512
    learning_rate = 0.001
    learning_rate_max_offset = 0.002
    fine_tuning_epochs = 2
    threshold = 0.31
    max_vocab_size = 120000
    embed_size = 300
    print_every_step = 500
    max_seq_len = 70
    share = True
    dropout = 0.1
    sub = pd.read_csv('./input/sample_submission.csv')
    train_df, test_df = load_data()
    is_training = False
    # shuffling
    trn_idx = np.random.permutation(len(train_df))
    train_df = train_df.iloc[trn_idx].reset_index(drop=True)
    n_folds = 5
    n_repeats = 1
    args = {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'threshold': threshold,
            'max_vocab_size': max_vocab_size,
            'embed_size': embed_size, 'print_every_step': print_every_step, 'dropout': dropout,
            'learning_rate_max_offset': learning_rate_max_offset,
            'fine_tuning_epochs': fine_tuning_epochs, 'max_seq_len': max_seq_len}
    predictions_te_all = np.zeros((len(test_df),))
    if is_training:
        for _ in range(n_repeats):
            if n_folds > 1:
                _, predictions_te, _, threshold, coeffs = cv(train_df, test_df, n_folds=n_folds, share=share, **args)
                print('coeff between predictions {}'.format(coeffs))
            else:
                predictions_te, _, _, _ = main(train_df, test_df, test_df, **args)
            predictions_te_all += predictions_te / n_repeats
        sub.prediction = predictions_te_all > threshold
        sub.to_csv("submission.csv", index=False)
    else:
        with open('question.txt') as file:
            question = file.readline()
            questionInput = [{'question_txt': clean_text(question)}]
            print(questionInput)
            # predictions_te, _, _, _ = main(train_df, test_df, test_df, **args)


