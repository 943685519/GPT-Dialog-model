import re
import os
import json
import random
import torch
import logging
from torch.utils.data import Dataset
import numpy as np
import argparse
from scipy.interpolate import RectBivariateSpline
from torch.utils.checkpoint import checkpoint
from collections import namedtuple, Counter
from attrdict import AttrDict

def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
>> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger




class Vocab:
    spl = '<p>'
    pad = '<pad>'
    eos = '</s>'
    unk = '<unk>'
    p1 = '<p1>'
    p2 = '<p2>'

    def __init__(self, vocab_file):
        # TODO: add check for special tokens
        self.spec_tokens = [Vocab.spl, Vocab.pad, Vocab.eos, Vocab.unk]
        with open(vocab_file, 'r', encoding='utf8') as fr:
            vocab = [line.strip('\n').split()[0] for line in fr.readlines()]
        vocab = self.spec_tokens + vocab
        # self.spec_tokens = [Vocab.spl, Vocab.pad, Vocab.eos, Vocab.unk, Vocab.p1, Vocab.p2]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[Vocab.pad]

    @property
    def spl_id(self):
        return self.token2id[Vocab.spl]

    @property
    def p1_id(self):
        return self.token2id[Vocab.p1]

    @property
    def p2_id(self):
        return self.token2id[Vocab.p2]

    @property
    def bos_id(self):
        return self.token2id[Vocab.eos]

    @property
    def eos_id(self):
        return self.token2id[Vocab.eos]

    def string2ids(self, string):
        tokens = string.split()
        ids = [self.token2id[t] for t in tokens if t in self.token2id]
        return ids

    def ids2string(self, ids):
        tokens = [self.id2token[id] for id in ids]
        return ''.join(tokens)

    def ids2string_wo_eos(self, ids):
        res = ''
        for id in ids[1:]:
            if id == self.eos_id:
                return res
            else:
                res += self.id2token[id]


class DialogDataset(Dataset):
    def __init__(self, paths, vocab, logger, max_lengths=2048):
        self.logger = logger
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = DialogDataset.make_dataset(paths, vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for line in lines:
                    # style, post, resp
                    dataset.append([int(line[0]),
                                    vocab.string2ids(' '.join(line[1].replace(' ', '')))[:max_lengths],
                                    vocab.string2ids(' '.join(line[2].replace(' ', '')))[:max_lengths]])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        style, post, resp = self.data[idx]
        post = [self.vocab.eos_id] + post + [self.vocab.eos_id]
        resp = [self.vocab.eos_id] + resp + [self.vocab.eos_id]
        return {"style": style, "post": post, "post_len": len(post), "resp": resp, "resp_len": len(resp)}


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['style'] = torch.LongTensor([i['style'] for i in batch])
        res['post_len'] = torch.LongTensor([i['post_len'] for i in batch])
        res['resp_len'] = torch.LongTensor([i['resp_len'] for i in batch])
        post_max_len = max([len(i['post']) for i in batch])
        resp_max_len = max([len(i['resp']) for i in batch])
        res['post'] = torch.LongTensor([i['post'] + [self.pad_id] * (post_max_len - len(i['post'])) for i in batch])
        res['resp'] = torch.LongTensor([i['resp'] + [self.pad_id] * (resp_max_len - len(i['resp'])) for i in batch])
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        return AttrDict(config)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def pad_sequence(sequences, batch_first=False, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def checkpoint_sequential(functions, segments, *inputs):
    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                inputs = functions[j](*inputs)
            return inputs

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)


def get_latest_ckpt(dir_name):
    files = [i for i in os.listdir(dir_name) if '.ckpt' in i]
    if len(files) == 0:
        return None
    else:
        res = ''
        num = -1
        for i in files:
            n = int(i.split('-')[-1].split('.')[0])
            if n > num:
                num = n
                res = i
        return res


def get_epoch_from_ckpt(ckpt):
    return int(ckpt.split('-')[-1].split('.')[0])


def get_ckpt_filename(name, epoch):
    return '{}-{}.ckpt'.format(name, epoch)


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)

    return scores


def openai_transformer_config():
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    cfg = dotdict({'n_layers': 12, 'n_embeddings': 40477, 'n_pos_embeddings': 512,
                   'embeddings_size': 768, 'n_heads': 12, 'dropout': 0.1,
                   'embed_dropout': 0.1, 'attn_dropout': 0.1, 'ff_dropout': 0.1})

    return cfg


def load_openai_weights_chinese(model, directory):
    openai_model = torch.load(directory)
    openai_model.pop('decoder.pre_softmax.weight')
    b = list(openai_model.keys())
    for i in b:
        openai_model['decoder.' + i] = openai_model.pop(i)
    model.load_state_dict(openai_model)


def load_openai_weights(model, directory, n_special_tokens=0):
    # TODO: add check of shapes

    parameters_names_path = os.path.join(directory, 'parameters_names.json')
    parameters_shapes_path = os.path.join(directory, 'parameters_shapes.json')
    parameters_weights_paths = [os.path.join(directory, 'params_{}.npy'.format(n)) for n in range(10)]

    with open(parameters_names_path, 'r') as parameters_names_file:
        parameters_names = json.load(parameters_names_file)

    with open(parameters_shapes_path, 'r') as parameters_shapes_file:
        parameters_shapes = json.load(parameters_shapes_file)

    parameters_weights = [np.load(path) for path in parameters_weights_paths]
    parameters_offsets = np.cumsum([np.prod(shape) for shape in parameters_shapes])
    parameters_weights = np.split(np.concatenate(parameters_weights, 0), parameters_offsets)[:-1]
    parameters_weights = [p.reshape(s) for p, s in zip(parameters_weights, parameters_shapes)]

    parameters_weights[1] = parameters_weights[1][1:]  # skip 0 - <unk>

    if model.pos_embeddings.num_embeddings - 1 > parameters_weights[0].shape[0]:
        xx = np.linspace(0, parameters_weights[0].shape[0], model.pos_embeddings.num_embeddings - 1)
        new_kernel = RectBivariateSpline(np.arange(parameters_weights[0].shape[0]),
                                         np.arange(parameters_weights[0].shape[1]),
                                         parameters_weights[0])
        parameters_weights[0] = new_kernel(xx, np.arange(parameters_weights[0].shape[1]))

    parameters_weights[0] = parameters_weights[0][:model.pos_embeddings.num_embeddings - 1]
    parameters_weights[1] = parameters_weights[1][:model.embeddings.num_embeddings - n_special_tokens]

    model.pos_embeddings.weight.data[1:] = torch.from_numpy(parameters_weights[0])
    model.embeddings.weight.data[n_special_tokens:] = torch.from_numpy(parameters_weights[1])

    parameters_weights = parameters_weights[2:]

    for name, weights in zip(parameters_names, parameters_weights):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ':0'
        name = name[:-2]
        name = name.split('/')

        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]

            pointer = getattr(pointer, l[0])

            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if len(weights.shape) == 3:  # conv1d to linear
            weights = weights[0].transpose((1, 0))

        pointer.data[...] = torch.from_numpy(weights)
