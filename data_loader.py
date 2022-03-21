import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
from pprint import pprint
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):
    # {'<pad>': 0, '<suc>': 1, 'name': 2, 'cont': 3, 'race': 4, 'title': 5, 'edu': 6, 'org': 7, 'pro': 8, 'loc': 9}
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []
  
    for index, instance in enumerate(data):
        # 以{'sentence': ['高', '勇', '：', '男', '，', '中', '国', '国', '籍', '，', '无', '境', '外', '居', '留', '权', '，'], 
        # 'ner': [{'index': [0, 1], 'type': 'NAME'}, 
        #      {'index': [5, 6, 7, 8], 'type': 'CONT'}], 
        # 'word': [[0, 1], [2], [3], [4], [5, 6], [7, 8], [9], [10], [11, 12], [13, 14, 15], [16]]}
        # 为例
        if len(instance['sentence']) == 0:
            continue
        # tokens:[['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']]
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        # 将字符转换为bert需要的token
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            # tokens:[['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']]
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                # 这里的start表示的是第i个token的起始位置
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        # _dist_inputs:
        """
        [[19 10 11 11 12 12 12 12 13 13 13 13 13 13 13 13 14]
        [ 1 19 10 11 11 12 12 12 12 13 13 13 13 13 13 13 13]
        [ 2  1 19 10 11 11 12 12 12 12 13 13 13 13 13 13 13]
        [ 2  2  1 19 10 11 11 12 12 12 12 13 13 13 13 13 13]
        [ 3  2  2  1 19 10 11 11 12 12 12 12 13 13 13 13 13]
        [ 3  3  2  2  1 19 10 11 11 12 12 12 12 13 13 13 13]
        [ 3  3  3  2  2  1 19 10 11 11 12 12 12 12 13 13 13]
        [ 3  3  3  3  2  2  1 19 10 11 11 12 12 12 12 13 13]
        [ 4  3  3  3  3  2  2  1 19 10 11 11 12 12 12 12 13]
        [ 4  4  3  3  3  3  2  2  1 19 10 11 11 12 12 12 12]
        [ 4  4  4  3  3  3  3  2  2  1 19 10 11 11 12 12 12]
        [ 4  4  4  4  3  3  3  3  2  2  1 19 10 11 11 12 12]
        [ 4  4  4  4  4  3  3  3  3  2  2  1 19 10 11 11 12]
        [ 4  4  4  4  4  4  3  3  3  3  2  2  1 19 10 11 11]
        [ 4  4  4  4  4  4  4  3  3  3  3  2  2  1 19 10 11]
        [ 4  4  4  4  4  4  4  4  3  3  3  3  2  2  1 19 10]
        [ 5  4  4  4  4  4  4  4  4  3  3  3  3  2  2  1 19]]
        """
        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])
        # _grid_labels：

        """
        [[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
        """
        # _entity_text：{'0-1-#-2', '5-6-7-8-#-3'}
        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return train_dataset, dev_dataset, test_dataset

def process_bert_predict(texts, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []
    for index, text in enumerate(texts):
        # 这里直接是以字为单位
        tokens = [tokenizer.tokenize(word) for word in text]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        # 将字符转换为bert需要的token
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(text)
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            # tokens:[['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']]
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                # 这里的start表示的是第i个token的起始位置
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)

    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts

def collate_fn_predict(data):
    bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts

class RelationDatasetPredict(Dataset):
    def __init__(self, bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts):
        self.bert_inputs = bert_inputs
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.texts = texts

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.texts[item]

    def __len__(self):
        return len(self.bert_inputs)

def load_data_bert_predict(texts, config):
  if isinstance(texts, str):
    texts = [texts]
  # with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
  #       train_data = json.load(f)
  # with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
  #     dev_data = json.load(f)
  # with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
  #     test_data = json.load(f)

  tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

  vocab = Vocabulary()
  # train_ent_num = fill_vocab(vocab, train_data)
  # dev_ent_num = fill_vocab(vocab, dev_data)
  # test_ent_num = fill_vocab(vocab, test_data)
  label2id = {'<pad>': 0, '<suc>': 1, 'name': 2, 'cont': 3, 'race': 4, 'title': 5, 'edu': 6, 'org': 7, 'pro': 8, 'loc': 9}
  id2label = {v:k for k,v in label2id.items()}
  vocab.label2id = label2id
  vocab.id2label = id2label
  print(dict(vocab.label2id))
  print("=============================")
  config.label_num = len(vocab.label2id)
  config.vocab = vocab
  print(config)
  # process_bert_predict(texts, tokenizer, vocab)
  predict_dataset = RelationDatasetPredict(*process_bert_predict(texts, tokenizer, vocab))
  return predict_dataset


if __name__ == "__main__":
  import config
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='./config/resume-zh.json')
  parser.add_argument('--device', type=int, default=0)

  parser.add_argument('--dist_emb_size', type=int)
  parser.add_argument('--type_emb_size', type=int)
  parser.add_argument('--lstm_hid_size', type=int)
  parser.add_argument('--conv_hid_size', type=int)
  parser.add_argument('--bert_hid_size', type=int)
  parser.add_argument('--ffnn_hid_size', type=int)
  parser.add_argument('--biaffine_size', type=int)

  parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

  parser.add_argument('--emb_dropout', type=float)
  parser.add_argument('--conv_dropout', type=float)
  parser.add_argument('--out_dropout', type=float)

  parser.add_argument('--epochs', type=int)
  parser.add_argument('--batch_size', type=int)

  parser.add_argument('--clip_grad_norm', type=float)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--weight_decay', type=float)

  parser.add_argument('--bert_name', type=str)
  parser.add_argument('--bert_learning_rate', type=float)
  parser.add_argument('--warm_factor', type=float)

  parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

  parser.add_argument('--seed', type=int)

  args = parser.parse_args()

  config = config.Config(args)
  """
  with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
    test_data = json.load(f)
  print(test_data[0])
  """
  texts = [
    "高勇，男，中国国籍，无境外居留权。"
  ]
  predict_dataset = load_data_bert_predict(texts, config)
  from torch.utils.data import DataLoader
  predict_loader = DataLoader(dataset=predict_dataset,
                   batch_size=config.batch_size,
                   collate_fn=collate_fn_predict,
                   shuffle=False,
                   num_workers=4,
                   drop_last=False)
  for i, data_batch in enumerate(predict_loader):
    bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
    print(bert_inputs)
    print(grid_mask2d)
    print(pieces2word)
    print(dist_inputs)
    print(sent_length)
                  