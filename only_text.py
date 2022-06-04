import argparse
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchcrf import CRF
from transformers import BertModel

from ner_evaluate import evaluate, evaluate_bdui
from tokenization import BertTokenizer
import seaborn as sns;sns.set_theme()
import matplotlib.pyplot as plt

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def load_word_matrix(args):
    if not os.path.exists(args.wordvec_dir):
        os.mkdir(args.wordvec_dir)

    # Making new word vector
    embedding_list = []
    embedding_list.append(np.random.uniform(-0.25, 0.25, args.word_emb_dim).tolist())
    embedding_list.append(np.random.uniform(-0.25, 0.25, args.word_emb_dim).tolist())
    embedding_list.append(np.random.uniform(-0.25, 0.25, args.word_emb_dim).tolist())
    embedding_list.append(np.random.uniform(-0.25, 0.25, args.word_emb_dim).tolist())
    word_index = dict()
    word_index['[pad]'] = 0
    word_index['[unk]'] = 1
    word_index['<number>'] = 2
    word_index['<url>'] = 3
    indix = 4
    with open(os.path.join(args.wordvec_dir, args.w2v_file), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0]
            word_index[word] = indix
            indix += 1
            embedding_list.append([float(x) for x in values[1:]])
    print("vocab_size:",len(embedding_list))
    args.word_vocab_size = len(embedding_list)

    return torch.tensor(embedding_list,dtype=torch.float32),word_index

def preprocess_word(word):
    number_re = r"[-+]?[.\d]*[\d]+[:,.\d]*"
    url_re = r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"
    hashtag_re = r"#\S+"
    user_re = r"@\w+"

    if re.compile(number_re).match(word):
        word = '<number>'
    elif re.compile(url_re).match(word):
        word = '<url>'
    elif re.compile(hashtag_re).match(word):
        word = word[1:]  # only erase `#` at the front
    elif re.compile(user_re).match(word):
        word = word[1:]  # only erase `@` at the front

    word = word.lower()

    return word

def get_examples(args,file,chars):
    with open(os.path.join(args.data_dir, file + '.txt'), 'r', encoding='utf-8') as f:
        sentences = []
        sentence = [[], []]
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.startswith("IMGID:"):
                if sentence[0]:
                    args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
                    sentences.append(sentence)
                    sentence = [[], []]
            else:
                word, tag = line.strip().split("\t")
                word = preprocess_word(word)
                args.max_word_len = max(args.max_word_len,len(word))
                for i in word:
                    if i not in chars:
                        chars.append(i)
                sentence[0].append(word)
                sentence[1].append(tag)
        if sentence[0]:
            args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
            sentences.append(sentence)
    return sentences,chars

def sentences_to_feature(args,sentences,word_id,char_id,label_vocab):
    word_input = torch.zeros((len(sentences),args.max_seq_len)).long()
    char_input = torch.zeros((len(sentences),args.max_seq_len,args.max_word_len)).long()
    #long 为softmax
    # mask_input = torch.zeros((len(sentences),args.max_seq_len)).long()
    #uint8 CRF
    mask_input = torch.zeros((len(sentences), args.max_seq_len),dtype=torch.uint8)
    labels = torch.zeros(len(sentences),args.max_seq_len).long()
    # labels = torch.zeros(len(sentences), args.max_seq_len,dtype=torch.uint8)

    for i,sen in enumerate(sentences):
        for j,word in enumerate(sen[0]):
            word_input[i][j] = word_id.get(word,1)
            mask_input[i][j] = 1
            for k,char in enumerate(word):
                char_input[i][j][k] = char_id.get(char,1)
        for j,label in enumerate(sen[1]):
            labels[i][j] = label_vocab.get(label)

    return TensorDataset(word_input,char_input,mask_input,labels)

def examples_convert(args,word_id,label_vocab):
    chars = []
    train_s,chars = get_examples(args,args.train_file,chars)
    dev_s,chars = get_examples(args,args.dev_file,chars)
    test_s, chars = get_examples(args,args.test_file, chars)
    print("max_seq_len:",args.max_seq_len,'max_word_len:',args.max_word_len)
    #word获取的是词向量所包含的，char获取的是整个数据集中出现的,根据设定的char_vocab_size截断多余的
    char_id = {}
    char_counts = Counter(chars)
    char_id["[pad]"] = 0
    char_id["[unk]"] = 1
    num = 2
    for i in char_counts.most_common():
        char_id[i[0]] = num
        num += 1
        if num==args.char_vocab_size:
            break

    #处理数据
    train_data = sentences_to_feature(args,train_s,word_id,char_id,label_vocab)
    dev_data = sentences_to_feature(args, dev_s, word_id, char_id,label_vocab)
    test_data = sentences_to_feature(args, test_s, word_id, char_id,label_vocab)

    return train_data,dev_data,test_data

class CNN_char(nn.Module):
    def __init__(self,args):
        super(CNN_char, self).__init__()
        self.char_embed = nn.Embedding(args.char_vocab_size,args.char_emb_dim,padding_idx=0)
        nn.init.uniform_(self.char_embed.weight,-0.25,0.25)
        self.cnns = nn.ModuleList([nn.Sequential(
            nn.Conv1d(args.char_emb_dim,args.num_filters,kernel_size=i,padding=i//2),
            nn.Tanh(),
            nn.MaxPool1d(args.max_word_len-i+1),
            nn.Dropout(0.4)
        )for i in args.kernel_lst])
        self.out = nn.Sequential(
            nn.Linear(args.num_filters*len(args.kernel_lst),128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,args.final_char_dim)
        )

    def forward(self,x):
        b,s,w = x.size()
        # b,s,w,d
        char_embed = self.char_embed(x)
        char_embed = char_embed.view(b*s,w,-1)
        #b*s,d,w
        char_embed = char_embed.permute(0,2,1)
        # b*s,num_filters*len(kernel)
        cnn = [cnn(char_embed).squeeze(-1) for cnn in self.cnns ]
        cnn = torch.cat(cnn,dim=-1)
        # b,s,final_dim
        out = self.out(cnn).view(b,s,-1)
        return out

class Word_char_lstm(nn.Module):
    def __init__(self,args,word_martix,labels_num=10):
        super(Word_char_lstm, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_martix)
        self.char_embed = CNN_char(args)
        self.lstm = nn.LSTM(input_size=args.final_char_dim+args.word_emb_dim,hidden_size=args.hidden_dim,bidirectional=True,batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2*args.hidden_dim,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,labels_num)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self,word_input,char_input):
        word_embed = self.word_embed(word_input)
        char_embed = self.char_embed(char_input)
        word_char_embed = torch.cat([word_embed,char_embed],dim=-1)
        lstm,_ = self.lstm(word_char_embed,None)
        out = self.out(lstm)
        return out

    def loss(self,out,label):
        return self.loss(out,label)

class Word_lstm(nn.Module):
    def __init__(self,args,word_martix,labels_num=10):
        super(Word_lstm, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_martix)
        self.lstm = nn.LSTM(input_size=args.word_emb_dim,hidden_size=args.hidden_dim,bidirectional=True,batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2*args.hidden_dim,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,labels_num)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self,word_input,char_input):
        word_embed = self.word_embed(word_input)
        lstm,_ = self.lstm(word_embed,None)
        out = self.out(lstm)
        return out

    def loss(self,out,label):
        return self.loss(out,label)

class Word_lstm_CRF(nn.Module):
    def __init__(self,args,word_martix,labels_num=12):
        super(Word_lstm_CRF, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_martix)
        self.lstm = nn.LSTM(input_size=args.word_emb_dim, hidden_size=args.hidden_dim,
                            bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2 * args.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_input,char_input,masks):
        word_embed = self.word_embed(word_input)
        lstm, _ = self.lstm(word_embed, None)
        out = self.out(lstm)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

class Word_char_lstm_CRF(nn.Module):
    def __init__(self,args,word_martix,labels_num=12):
        super(Word_char_lstm_CRF, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_martix)
        self.char_embed = CNN_char(args)
        self.lstm = nn.LSTM(input_size=args.final_char_dim + args.word_emb_dim, hidden_size=args.hidden_dim,
                            bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2 * args.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_input,char_input,masks):
        word_embed = self.word_embed(word_input)
        char_embed = self.char_embed(char_input)
        word_char_embed = torch.cat([word_embed, char_embed], dim=-1)
        lstm, _ = self.lstm(word_char_embed, None)
        out = self.out(lstm)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

def train(args,train_data,dev_data,test_data,word_embed,label_vocab):
    train_iter = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=args.eval_batch_size)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.eval_batch_size)

    #word char 都有
    model = Word_char_lstm(args,word_embed)
    #word 只有
    # model = Word_lstm(args,word_embed)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    for epoch in range(args.epochs):
        for batch in train_iter:
            train_w_ids_t,train_c_ids_t, train_mask_t, train_labels_t = batch
            train_w_ids_t,train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(),train_c_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()

            #LSTM
            out = model(train_w_ids_t,train_c_ids_t)
            #CRF
            # out = model(train_w_ids_t,train_c_ids_t,train_mask_t)
            # print(out[:20])
            loss = F.cross_entropy(out.view(-1,10), train_labels_t.view(-1),reduction='none')
            loss = loss*train_mask_t.view(-1)
            loss = loss.mean()
            # print("loss:",loss)
            # print(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                preds,labels = [],[]
                for batch in dev_iter:
                    train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = batch
                    train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_c_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()

                    #LSTM
                    out = model(train_w_ids_t, train_c_ids_t)
                    #BERT
                    # out = model(train_w_ids_t, train_c_ids_t, train_mask_t)
                    pred = out.argmax(axis=-1)
                    pred = pred*train_mask_t
                    for i in pred.long():
                        preds.append([j.item() for j in i if j.item()!=0])
                    for i in train_labels_t.long():
                        labels.append([j.item() for j in i if j.item() != 0])
                wordss = torch.zeros((len(preds),50)).long()
                acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                print("dev准确率:",acc,"F1:",f1)
                if max_dev_f1<f1:
                    max_dev_f1 = f1
                    preds, labels = [],[]
                    for batch in test_iter:
                        train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = batch
                        train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_c_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()
                        #LSTM
                        out = model(train_w_ids_t, train_c_ids_t)
                        #BERT
                        # out = model(train_w_ids_t, train_c_ids_t, train_mask_t)
                        pred = out.argmax(axis=-1)
                        pred = pred * train_mask_t
                        for i in pred.long():
                            preds.append([j.item() for j in i if j.item() != 0])
                        for i in train_labels_t.long():
                            labels.append([j.item() for j in i if j.item() != 0])
                    wordss = torch.zeros((len(preds), 50)).long()
                    acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                    print("test准确率:",acc,"F1:",f1)
            model.train()
    return 0

def train_CRF(args,train_data,dev_data,test_data,word_embed,label_vocab):
    #lstm
    train_iter = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=args.eval_batch_size)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.eval_batch_size)

    #word char 都有
    model = Word_char_lstm_CRF(args,word_embed)
    #word
    # model = Word_lstm_CRF(args,word_embed)

    model.to(torch.device('cuda:0'))
    #LSTM
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    for epoch in range(args.epochs):
        for batch in train_iter:
            train_w_ids_t,train_c_ids_t, train_mask_t, train_labels_t = batch
            train_w_ids_t,train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(),train_c_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

            #LSTM
            # out = model(train_w_ids_t,train_c_ids_t)
            # CRF
            out = model(train_w_ids_t, train_c_ids_t, train_mask_t)
            loss = model.loss_fn(out=out, tags=train_labels_t, masks=train_mask_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                preds,labels = [],[]
                for batch in dev_iter:
                    train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = batch
                    train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_c_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                    # LSTM
                    # out = model(train_w_ids_t,train_c_ids_t)
                    # CRF
                    out = model(train_w_ids_t, train_c_ids_t, train_mask_t)
                    pred = model.predict(out,train_mask_t)
                    preds += pred
                    for i in train_labels_t.long():
                        labels.append([j.item() for j in i if j.item() != 0])
                wordss = torch.zeros((len(preds),50)).long()
                acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                print("dev准确率:",acc,"F1:",f1)
                if max_dev_f1<f1:
                    max_dev_f1 = f1
                    preds, labels = [],[]
                    for batch in test_iter:
                        train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = batch
                        train_w_ids_t, train_c_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_c_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                        # LSTM
                        # out = model(train_w_ids_t,train_c_ids_t)
                        # CRF
                        out = model(train_w_ids_t, train_c_ids_t, train_mask_t)
                        pred = model.predict(out, train_mask_t)
                        preds += pred
                        for i in train_labels_t.long():
                            labels.append([j.item() for j in i if j.item() != 0])
                    wordss = torch.zeros((len(preds), 50)).long()
                    acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                    print("test准确率:",acc,"F1:",f1)
            model.train()
    return 0




#BERT
# def fine_grade_tokenize(texts, tokenizer):
#     """
#     BERT遇到空格等不会放进去
#     """
#     for i,line in enumerate(texts):
#         for j,ch in enumerate(line[0]):
#             # print(tokenizer.tokenize(ch))
#             if not len(tokenizer.tokenize(ch)):
#                 texts[i][0][j] = '[UNK]'
#             # if ch in [' ', '\t', '\n']:
#             #     # print(ch)
#             #     texts[i][0][j] = '[PAD]'
#             # elif not len(tokenizer.tokenize(ch)):
#             #     # print("len no",texts[i][0][j])
#             #     texts[i][0][j] = '[PAD]'
#     # '<number>' UNK
#     return texts

# def preprocess_word_BERT(word):
#     number_re = r"[-+]?[.\d]*[\d]+[:,.\d]*"
#     url_re = r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"
#     hashtag_re = r"#\S+"
#     user_re = r"@\w+"
#
#     if re.compile(number_re).match(word):
#         word = '<number>'
#     elif re.compile(url_re).match(word):
#         word = '<url>'
#     elif re.compile(hashtag_re).match(word):
#         word = word[1:]  # only erase `#` at the front
#     elif re.compile(user_re).match(word):
#         word = word[1:]  # only erase `@` at the front
#
#     word = word.lower()
#
#     return word

def get_examples_BERT(args,file,tokenizer):
    with open(os.path.join(args.data_dir, file + '.txt'), 'r', encoding='utf-8') as f:
        sentences = []
        sentence = [[], []]
        word_one = []
        words = []
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.startswith("IMGID:"):
                if sentence[0]:
                    sentence[0] = ['[CLS]']+sentence[0]+['[SEP]']
                    sentence[1] = ['O']+sentence[1]+['O']
                    args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
                    sentences.append(sentence)
                    sentence = [[], []]
                    word_one = ['[CLS]']+word_one+['[SEP]']
                    words.append(word_one)
                    word_one = []
            else:
                word, tag = line.strip().split("\t")
                #case不要,uncase要
                if 'uncase' in args.model_path:
                    word = word.lower()
                word_one.append(word)
                # word = word.lower()
                token = tokenizer.tokenize(word)
                if len(token) == 0:
                    token = ['[UNK]']
                for m in range(len(token)):
                    if m == 0:
                        sentence[1].append(tag)
                    else:
                        sentence[1].append("X")
                args.max_word_len = max(args.max_word_len,len(word))
                sentence[0] += token
        if sentence[0]:
            sentence[0] = ['[CLS]'] + sentence[0] + ['[SEP]']
            sentence[1] = ['O'] + sentence[1] + ['O']
            args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
            sentences.append(sentence)
            word_one = ['[CLS]'] + word_one + ['[SEP]']
            words.append(word_one)
    return sentences,words

def get_examples_BERT_special(args,file,tokenizer):
    #词前后加入特殊字符
    with open(os.path.join(args.data_dir, file + '.txt'), 'r', encoding='utf-8') as f:
        sentences = []
        sentence = [[], []]
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.startswith("IMGID:"):
                if sentence[0]:
                    sentence[0] = ['[CLS]']+sentence[0]+['[SEP]']
                    sentence[1] = ['O']+sentence[1]+['O']
                    args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
                    sentences.append(sentence)
                    sentence = [[], []]
            else:
                word, tag = line.strip().split("\t")
                word = word.lower()
                token = tokenizer.tokenize(word)
                if len(token) == 0:
                    token = ['[UNK]']
                token = ['[unused1]'] + token + ['[unused2]']
                for m in range(len(token)):
                    if m == 0:
                        sentence[1].append(tag)
                    else:
                        sentence[1].append("X")
                args.max_word_len = max(args.max_word_len,len(word))
                sentence[0] += token
        if sentence[0]:
            sentence[0] = ['[CLS]'] + sentence[0] + ['[SEP]']
            sentence[1] = ['O'] + sentence[1] + ['O']
            args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
            sentences.append(sentence)
    return sentences

def sentences_to_feature_BERT(args,sentences,tokenizer,label_vocab):
    word_input = torch.zeros((len(sentences),args.max_seq_len)).long()
    #long 为softmax
    # mask_input = torch.zeros((len(sentences),args.max_seq_len)).long()
    #uint8 CRF
    mask_input = torch.zeros((len(sentences), args.max_seq_len),dtype=torch.uint8)
    labels = torch.zeros(len(sentences),args.max_seq_len).long()
    # labels = torch.zeros(len(sentences), args.max_seq_len,dtype=torch.uint8)

    for i,sen in enumerate(sentences):
        word_input[i][:len(sen[0])] = torch.tensor(tokenizer.convert_tokens_to_ids(sen[0]))
        for j,word in enumerate(sen[0]):
            mask_input[i][j] = 1
        for j,label in enumerate(sen[1]):
            labels[i][j] = label_vocab.get(label)
    # for i in range(5):
    #     print(sentences[i][0])
    #     print(word_input[i])
    #     print(labels[i])
    return TensorDataset(word_input,mask_input,labels)


def examples_convert_BERT(args,label_vocab,tokenizer,flag=0):
    words = []
    if flag==0:
        train_s,_ = get_examples_BERT(args,args.train_file,tokenizer)
        dev_s,_ = get_examples_BERT(args,args.dev_file,tokenizer)
        test_s,words = get_examples_BERT(args,args.test_file,tokenizer)
    else:
        train_s = get_examples_BERT_special(args, args.train_file, tokenizer)
        dev_s = get_examples_BERT_special(args, args.dev_file, tokenizer)
        test_s = get_examples_BERT_special(args, args.test_file, tokenizer)
        for i in test_s:
            words.append(i[0])

    print("max_seq_len:",args.max_seq_len,'max_word_len:',args.max_word_len)
    #word获取的是词向量所包含的，char获取的是整个数据集中出现的,根据设定的char_vocab_size截断多余的
    #处理数据
    train_data = sentences_to_feature_BERT(args,train_s,tokenizer,label_vocab)
    dev_data = sentences_to_feature_BERT(args, dev_s, tokenizer, label_vocab)
    test_data = sentences_to_feature_BERT(args, test_s, tokenizer,label_vocab)

    return train_data,dev_data,test_data,words

class Word_BERT(nn.Module):
    def __init__(self,args,labels_num=11):
        super(Word_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_path)
        # self.bert_config = self.bert.config
        self.out = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,labels_num)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self,word_input,masks):
        # with torch.no_grad():
        sequence_output = self.bert(word_input, attention_mask=masks).last_hidden_state
        out = self.out(sequence_output)
        return out

    def loss(self,out,label):
        return self.loss(out,label)

class BERT_CRF(nn.Module):
    def __init__(self,args,labels_num=11):
        super(BERT_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_path)
        self.out = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_input,masks):
        sequence_output = self.bert(word_input, attention_mask=masks).last_hidden_state
        out = self.out(sequence_output)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

def train_BERT(args,train_data,dev_data,test_data,label_vocab):
    train_iter = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=args.eval_batch_size)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.eval_batch_size)

    # BERT版本
    model = Word_BERT(args)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    for epoch in range(args.epochs):
        for batch in train_iter:
            train_w_ids_t,train_mask_t, train_labels_t = batch
            train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()

            #BERT
            out = model(train_w_ids_t,train_mask_t)
            loss = F.cross_entropy(out.view(-1,11), train_labels_t.view(-1),reduction='none')
            loss = loss*train_mask_t.view(-1)
            loss = loss.mean()
            # print("loss:",loss)
            # print(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                preds,labels = [],[]
                for batch in dev_iter:
                    train_w_ids_t, train_mask_t, train_labels_t = batch
                    train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()

                    out = model(train_w_ids_t,train_mask_t)
                    pred = out.argmax(axis=-1)
                    pred = pred*train_mask_t
                    pred = pred.long()
                    for i,m in enumerate(train_labels_t.long()):
                        t1 = []
                        t2 = []
                        for j,_ in enumerate(train_labels_t[i]):
                            if train_labels_t[i][j].item() not in [0,10]:
                                t1.append(pred[i][j].item())
                                t2.append(train_labels_t[i][j].item())
                        preds.append(t1)
                        labels.append(t2)
                wordss = torch.zeros((len(preds),50)).long()
                acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                print("dev准确率:",acc,"F1:",f1)
                if max_dev_f1<f1:
                    max_dev_f1 = f1
                    preds, labels = [],[]
                    for batch in test_iter:
                        train_w_ids_t, train_mask_t, train_labels_t = batch
                        train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.float().cuda(), train_labels_t.cuda()

                        out = model(train_w_ids_t, train_mask_t)
                        pred = out.argmax(axis=-1)
                        pred = pred * train_mask_t
                        pred = pred.long()
                        for i, m in enumerate(train_labels_t.long()):
                            t1 = []
                            t2 = []
                            for j, _ in enumerate(train_labels_t[i]):
                                if train_labels_t[i][j].item() not in [0, 10]:
                                    t1.append(pred[i][j].item())
                                    t2.append(train_labels_t[i][j].item())
                            preds.append(t1)
                            labels.append(t2)
                    wordss = torch.zeros((len(preds), 50)).long()
                    acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                    print("test准确率:",acc,"F1:",f1)
            model.train()
    return 0

def train_CRF_BERT(args,train_data,dev_data,test_data,label_vocab):
    train_iter = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=args.eval_batch_size)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.eval_batch_size)

    #BERT版本
    model = BERT_CRF(args)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.00001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    max_preds,max_labels = None,None
    for epoch in range(args.epochs):
        for batch in train_iter:
            train_w_ids_t, train_mask_t, train_labels_t = batch
            train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

            # BERT
            out = model(train_w_ids_t, train_mask_t)
            loss = model.loss_fn(out=out, tags=train_labels_t, masks=train_mask_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                preds,labels = [],[]
                for batch in dev_iter:
                    train_w_ids_t, train_mask_t, train_labels_t = batch
                    train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                    # BERT
                    out = model(train_w_ids_t, train_mask_t)
                    pred = model.predict(out,train_mask_t)
                    for i, m in enumerate(train_labels_t.long()):
                        t1 = []
                        t2 = []
                        for j, _ in enumerate(train_labels_t[i]):
                            if train_labels_t[i][j].item() not in [0, 10]:
                                t1.append(pred[i][j])
                                t2.append(train_labels_t[i][j].item())
                        preds.append(t1)
                        labels.append(t2)
                wordss = torch.zeros((len(preds),50)).long()
                acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                print("epoch:",epoch,"dev准确率:",acc,"F1:",f1)
                if max_dev_f1<f1:
                    max_dev_f1 = f1
                    preds, labels = [],[]
                    l_preds,l_labels = [],[]
                    for batch in test_iter:
                        train_w_ids_t, train_mask_t, train_labels_t = batch
                        train_w_ids_t, train_mask_t, train_labels_t = train_w_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()
                        # BERT
                        out = model(train_w_ids_t, train_mask_t)
                        pred = model.predict(out, train_mask_t)
                        for i, m in enumerate(train_labels_t.long()):
                            t1 = []
                            t2 = []
                            l_t1,l_t2 = [],[]
                            for j, _ in enumerate(train_labels_t[i]):
                                if train_labels_t[i][j].item() not in [0]:
                                    l_t1.append(pred[i][j])
                                    l_t2.append(train_labels_t[i][j].item())
                                if train_labels_t[i][j].item() not in [0, 10]:
                                    t1.append(pred[i][j])
                                    t2.append(train_labels_t[i][j].item())
                            preds.append(t1)
                            labels.append(t2)
                            l_preds.append(l_t1)
                            l_labels.append(l_t2)
                    wordss = torch.zeros((len(preds), 50)).long()
                    acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                    print("test准确率:",acc,"F1:",f1)
                    max_preds = preds
                    max_labels = labels
            model.train()
    return max_preds,max_labels,l_preds,l_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/twitter2017", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--wordvec_dir", default="./wordvec", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)

    parser.add_argument("--train_file", default="train", type=str, help="Train file")
    parser.add_argument("--dev_file", default="valid", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test", type=str, help="Test file")
    parser.add_argument("--w2v_file", default="word_vector_200d.vec", type=str, help="Pretrained word vector file")
    parser.add_argument("--img_feature_file", default="img_vgg_features.pt", type=str,help="Filename for preprocessed image features")

    parser.add_argument("--max_seq_len", default=52, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=30, type=int, help="Max word length")

    parser.add_argument("--word_vocab_size", default=23204, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=102, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=200, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=32, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=64, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=128, type=int,help="Dimension of BiLSTM output, att layer (denoted as k) etc.")
    parser.add_argument("--before_att_size", default=64, type=int, help="Dimension of character cnn output")
    parser.add_argument("--kernel_lst", default=[2,3,4], type=list, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--epochs", default=30, type=int, help="")

    parser.add_argument("--model_path", default='D:\\transformerFileDownload\\Pytorch\\bert-base-uncased', type=str, help="")
    # parser.add_argument("--optimizer", default="adam", type=str,help="Optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
    # parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate")
    # parser.add_argument("--num_train_epochs", default=5.0, type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--slot_pad_label", default="[pad]", type=str,
    #                     help="Pad token for slot label pad (to be ignore when calculate loss)")
    # parser.add_argument("--ignore_index", default=0, type=int,
    #                     help='Specifies a target value that is ignored and does not contribute to the input gradient')
    args = parser.parse_args()

    set_seed(args)
    word_embed,word_id = load_word_matrix(args)
    label_vocab = {"[pad]": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7,"I-MISC": 8, "O": 9,"X":10}
    id_label = {i: j for j, i in label_vocab.items()}

    #LSTM
    # train_data, dev_data, test_data = examples_convert(args,word_id,label_vocab)
    # train(args,train_data,dev_data,test_data,word_embed,label_vocab)
    # train_CRF(args, train_data, dev_data, test_data, word_embed, label_vocab)

    #BERT
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_data, dev_data, test_data,_ = examples_convert_BERT(args,label_vocab,tokenizer)
    train_BERT(args, train_data, dev_data, test_data, label_vocab)
    # train_CRF_BERT(args, train_data, dev_data, test_data, label_vocab)

    # train_data, dev_data, test_data,words = examples_convert_BERT(args, label_vocab, tokenizer,0)
    # normal_preds,n_l = train_CRF_BERT(args, train_data, dev_data, test_data, label_vocab)
    # with open('./wc_preds.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in normal_preds:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)
    # with open('./wc_labels.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in n_l:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)

    # train_data, dev_data, test_data,s_words = examples_convert_BERT(args, label_vocab, tokenizer, 1)
    # s_preds, s_l,l_preds,l_labels = train_CRF_BERT(args, train_data, dev_data, test_data, label_vocab)
    #
    # with open('./wci_preds.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in s_preds:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)
    # with open('./wci_labels.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in s_l:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)
    # with open('./3_preds.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in l_preds:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)
    # with open('./3_labels.txt','w',encoding='utf-8') as f:
    #     preds_n = []
    #     for i in l_labels:
    #         strr = ""
    #         for j,k in enumerate(i):
    #             if j != len(i)-1:
    #                 strr += str(k)+' '
    #             else:
    #                 strr += str(k)+'\n'
    #         preds_n.append(strr)
    #     f.writelines(preds_n)

    #不同模型输出结果测评
    # n_preds,labels,s_preds,preds_3,labels_3 = [],[],[],[],[]
    # with open('./wc_preds.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         i = i.strip()
    #         pred = i.split(' ')
    #         pred = [int(j) for j in pred]
    #         n_preds.append(pred)
    # with open('./wci_preds.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         i = i.strip()
    #         pred = i.split(' ')
    #         pred = [int(j) for j in pred]
    #         s_preds.append(pred)
    # with open('./wc_labels.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         i = i.strip()
    #         pred = i.split(' ')
    #         pred = [int(j) for j in pred]
    #         labels.append(pred)
    # with open('./3_preds.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         i = i.strip()
    #         pred = i.split(' ')
    #         pred = [int(j) for j in pred]
    #         preds_3.append(pred)
    # with open('./3_labels.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         i = i.strip()
    #         pred = i.split(' ')
    #         pred = [int(j) for j in pred]
    #         labels_3.append(pred)
    #
    # evaluate_bdui(n_preds, s_preds, labels, words, label_vocab, id_label,preds_3,labels_3,s_words)

