import argparse
import os
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchcrf import CRF

from ner_evaluate import evaluate, evaluate_bdui2
from only_text import set_seed, load_word_matrix, preprocess_word
from resnet import Myresnet
from tokenization import BertTokenizer
from torchvision import transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from resnet import *
import resnet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;sns.set_theme()

#数据处理
def get_examples_BERT(args,file,chars):
    with open(os.path.join(args.data_dir, file + '.txt'), 'r', encoding='utf-8') as f:
        sentences = []
        sentence = [[], []]
        imgs = []
        imgid = None

        word_one = []
        words = []
        for line in f:
            line = line.strip()
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1] + '.jpg'
                continue
            if line == "":
                if len(sentence[1])>0:
                    args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
                    imgs.append(imgid)
                    sentences.append(sentence)
                    sentence = [[], []]
                    imgid = None

                    words.append(word_one)
                    word_one = []
            else:
                word, tag = line.strip().split("\t")

                word_one.append(word)

                word = preprocess_word(word)
                args.max_word_len = max(args.max_word_len, len(word))
                for i in word:
                    if i not in chars:
                        chars.append(i)
                sentence[0].append(word)
                sentence[1].append(tag)
        if sentence[0]:
            args.max_seq_len = max(args.max_seq_len, len(sentence[1]))
            sentences.append(sentence)

            words.append(word_one)
    return sentences,chars,imgs,words

def img_process(args,imgs,transform):
    image_path = os.path.join(args.image_dir,imgs)
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def sentences_to_feature_BERT(args,sentences,imgs,word_id,char_id,label_vocab):
    transform = transforms.Compose([
        # transforms.RandomCrop(224),  # args.crop_size, by default it is set to be 224
        # transforms.RandomResizedCrop()
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    sen_len = len(sentences)
    word_input = torch.zeros((sen_len,args.max_seq_len)).long()
    char_input = torch.zeros((sen_len, args.max_seq_len, args.max_word_len)).long()
    char_mask = torch.zeros((sen_len,args.max_seq_len)).long()
    #uint8 CRF
    mask_input = torch.zeros((sen_len, args.max_seq_len),dtype=torch.uint8)
    labels = torch.zeros(sen_len,args.max_seq_len).long()
    img_input = torch.zeros(sen_len,3,224,224)
    count = 0
    for i in range(sen_len):
        try:
            img_input[i] = img_process(args,imgs[i],transform)
        except:
            # print(imgs[i])
            count+=1
            img_input[i] = img_process(args,'17_06_4705.jpg',transform)
    print("not found:",count)
    #train count 362 ; sentences 3373

    for i, sen in enumerate(sentences):
        for j, word in enumerate(sen[0]):
            word_input[i][j] = word_id.get(word, 1)
            mask_input[i][j] = 1
            char_mask[i][j] = len(word)-1
            for k, char in enumerate(word):
                #lstm,cnn char输入都是一样的
                char_input[i][j][k] = char_id.get(char, 1)
        for j, label in enumerate(sen[1]):
            labels[i][j] = label_vocab.get(label)
    return TensorDataset(word_input,char_input,char_mask,mask_input,labels,img_input)

def examples_convert_BERT(args,label_vocab,word_id):
    chars = []
    train_s,chars,train_imgs,_ = get_examples_BERT(args,args.train_file,chars)
    dev_s,chars,dev_imgs,_ = get_examples_BERT(args,args.dev_file,chars)
    test_s,chars,test_imgs,words = get_examples_BERT(args,args.test_file,chars)

    print("max_seq_len:",args.max_seq_len,'max_word_len:',args.max_word_len)
    char_id = {}
    char_counts = Counter(chars)
    char_id["[pad]"] = 0
    char_id["[unk]"] = 1
    num = 2
    for i in char_counts.most_common():
        char_id[i[0]] = num
        num += 1
        if num == args.char_vocab_size:
            break
    #处理数据
    train_data = sentences_to_feature_BERT(args,train_s,train_imgs,word_id,char_id,label_vocab)
    dev_data = sentences_to_feature_BERT(args, dev_s, dev_imgs,word_id,char_id, label_vocab)
    test_data = sentences_to_feature_BERT(args, test_s, test_imgs,word_id,char_id,label_vocab)

    return train_data,dev_data,test_data,words

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
            nn.Linear(128,args.before_att_size)
        )

    def forward(self,x,y):
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

class LSTM_char(nn.Module):
    def __init__(self,args):
        super(LSTM_char, self).__init__()
        self.char_embed = nn.Embedding(args.char_vocab_size,args.char_emb_dim,padding_idx=0)
        nn.init.uniform_(self.char_embed.weight,-0.25,0.25)
        self.lstm = nn.LSTM(input_size=args.char_emb_dim,hidden_size=args.char_lstm_hidden_size,bidirectional=True,batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2 * args.char_lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.before_att_size)
        )

    def forward(self,x,char_mask):
        b,s,w = x.size()
        # b,s,w,d
        char_embed = self.char_embed(x)
        char_embed = char_embed.view(b*s,w,-1)
        #b*s,w,d
        char_lstm,_ = self.lstm(char_embed,None)

        char_mask = char_mask.view(b*s,1,1)
        char_mask = char_mask.expand(b*s,1,char_lstm.size()[-1])
        char_lstm = char_lstm.gather(1,char_mask).view(b,s,-1)
        out = self.out(char_lstm)
        return out

#模型
class W_C_I_anttention(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_I_anttention, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_embed = image_embed
        self.image_before_att = nn.Linear(2048,args.before_att_size)
        self.word_before_att = nn.Linear(args.word_emb_dim,args.before_att_size)
        self.att_w = nn.Sequential(
            nn.Linear(args.before_att_size,args.before_att_size),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, labels_num)
        )
        self.lstm = nn.LSTM(input_size=args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask):
        # b,2048
        img_embed = self.image_embed(img_id)
        # b,256
        img_embed = self.image_before_att(img_embed)
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)
        # b,s,256
        word_embed = self.word_before_att(word_embed)

        b,s,_ = word_embed.size()
        img_embed = img_embed.unsqueeze(1).expand(b,s,-1).contiguous().view(b*s,1,-1)
        char_embed = char_embed.view(b*s,1,-1)
        word_embed = word_embed.view(b*s,1,-1)
        # b*s,3,256
        embed = torch.cat([word_embed,char_embed,img_embed],dim=1)
        # b*s,256,3
        embed_att = self.att_w(embed).permute(0,2,1)
        embed_att = self.softmax(embed_att)

        embed_att = embed_att.contiguous().view(b*s,-1,1,3)
        embed = embed.permute(0,2,1).contiguous().view(b*s,-1,3,1)
        # b,s,256
        multi = torch.matmul(embed_att,embed).view(b,s,-1)
        #加不加lstm
        multi,_ = self.lstm(multi,None)
        out = self.out(multi)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

class W_C_anttention(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_anttention, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_before_att = nn.Linear(2048,args.before_att_size)
        self.word_before_att = nn.Linear(args.word_emb_dim,args.before_att_size)
        self.att_w = nn.Sequential(
            nn.Linear(args.before_att_size,args.before_att_size),
            nn.Dropout(0.4),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, labels_num)
        )
        self.lstm = nn.LSTM(input_size=args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask):
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)
        # b,s,256
        word_embed = self.word_before_att(word_embed)
        b,s,_ = word_embed.size()

        #attention
        char_embed = char_embed.view(b*s,1,-1)
        word_embed = word_embed.view(b*s,1,-1)
        # # b*s,2,256
        embed = torch.cat([word_embed,char_embed],dim=1)
        # b*s,256,2
        embed_att = self.att_w(embed).permute(0,2,1)
        embed_att = self.softmax(embed_att)

        embed_att = embed_att.contiguous().view(b*s,-1,1,2)
        embed = embed.permute(0,2,1).contiguous().view(b*s,-1,2,1)
        # b,s,256
        multi = torch.matmul(embed_att,embed).view(b,s,-1)
        multi,_ = self.lstm(multi,None)
        # 不加attention
        # embed = torch.cat([word_embed,char_embed],dim=-1)
        # multi, _ = self.lstm(embed, None)

        out = self.out(multi)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

class W_C_I_anttention_cheng(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_I_anttention_cheng, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_embed = image_embed
        self.image_before_att = nn.Linear(2048,args.before_att_size)
        self.word_before_att = nn.Linear(args.word_emb_dim,args.before_att_size)
        self.att_w = nn.Sequential(
            nn.Linear(args.before_att_size,args.before_att_size),
            nn.Dropout(0.2),
            # nn.Sigmoid()
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, labels_num)
        )
        self.lstm = nn.LSTM(input_size=args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask):
        # b,2048
        img_embed = self.image_embed(img_id)
        # b,256
        img_embed = self.image_before_att(img_embed)
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)
        # b,s,256
        word_embed = self.word_before_att(word_embed)

        b,s,_ = word_embed.size()
        img_embed = img_embed.unsqueeze(1).expand(b,s,-1).contiguous().view(b*s,1,-1)
        char_embed = char_embed.view(b*s,1,-1)
        word_embed = word_embed.view(b*s,1,-1)
        # b*s,3,256
        embed = torch.cat([word_embed,char_embed,img_embed],dim=1)
        # b*s,256,3
        embed_att = self.att_w(embed).permute(0,2,1)
        embed_att = self.softmax(embed_att)

        embed_att = embed_att.contiguous().view(b*s,-1,1,3)
        embed = embed.permute(0,2,1).contiguous().view(b*s,-1,3,1)
        # b,s,256
        multi = torch.matmul(embed_att,embed).view(b,s,-1)
        #加不加lstm
        multi,_ = self.lstm(multi,None)
        out = self.out(multi)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

def train_anttention_CRF(args,train_data,dev_data,test_data,word_embed,label_vocab):
    train_iter = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=args.eval_batch_size)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.eval_batch_size)

    # BERT版本
    char_embed = LSTM_char(args)
    # char_embed = CNN_char(args)
    # resnet50 = resnet.resnet50()
    # resnet50.load_state_dict(torch.load(os.path.join(args.resnet50_dir,'resnet50-19c8e357.pth')), strict=True)
    # my_resnet = Myresnet(resnet50)
    #152
    resnet152 = resnet.resnet152()
    resnet152.load_state_dict(torch.load(os.path.join(args.resnet152_dir, 'resnet152-b121ed2d.pth')), strict=True)
    my_resnet = Myresnet(resnet152)
    #
    model = W_C_I_anttention(args,word_embed,char_embed,my_resnet)
    # model = W_C_anttention(args,word_embed,char_embed,my_resnet)
    model.to(torch.device('cuda:0'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.00001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    max_preds, max_labels = None, None
    for epoch in range(args.epochs):
        for batch in train_iter:
            word_input,char_input,char_mask,train_mask_t,train_labels_t,img_input = batch
            word_input,char_input,char_mask,train_mask_t,train_labels_t,img_input = word_input.cuda(),char_input.cuda(),char_mask.cuda(),train_mask_t.cuda(),train_labels_t.cuda(),img_input.cuda()

            out = model(word_input,char_input,img_input,char_mask)
            loss = model.loss_fn(out=out, tags=train_labels_t, masks=train_mask_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():

                # preds, labels = [], []
                # for batch in train_iter:
                #     word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = batch
                #     word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = word_input.cuda(), char_input.cuda(), char_mask.cuda(), train_mask_t.cuda(), train_labels_t.cuda(), img_input.cuda()
                #
                #     out = model(word_input, char_input, img_input, char_mask)
                #     pred = model.predict(out, train_mask_t)
                #     for i, m in enumerate(train_labels_t.long()):
                #         t1 = []
                #         t2 = []
                #         for j, _ in enumerate(train_labels_t[i]):
                #             if train_labels_t[i][j].item() not in [0, 10]:
                #                 t1.append(pred[i][j])
                #                 t2.append(train_labels_t[i][j].item())
                #         preds.append(t1)
                #         labels.append(t2)
                # wordss = torch.zeros((len(preds), 50)).long()
                # acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                # print("train准确率:", acc, "F1:", f1)

                preds,labels = [],[]
                for batch in dev_iter:
                    word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = batch
                    word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = word_input.cuda(), char_input.cuda(), char_mask.cuda(), train_mask_t.cuda(), train_labels_t.cuda(), img_input.cuda()

                    out = model(word_input, char_input, img_input, char_mask)
                    pred = model.predict(out, train_mask_t)
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
                print("dev准确率:",acc,"F1:",f1)
                if max_dev_f1<f1:
                    max_dev_f1 = f1
                    preds, labels = [],[]
                    l_preds, l_labels = [], []
                    for batch in test_iter:
                        word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = batch
                        word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input = word_input.cuda(), char_input.cuda(), char_mask.cuda(), train_mask_t.cuda(), train_labels_t.cuda(), img_input.cuda()

                        out = model(word_input, char_input, img_input, char_mask)
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
    parser.add_argument("--resnet50_dir",default="D:\\transformerFileDownload\\Pytorch\\resnet50",help="Path for pretrained resnet50")
    parser.add_argument("--resnet152_dir", default="D:\\transformerFileDownload\\Pytorch\\resnet152",help="Path for pretrained resnet152")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)
    parser.add_argument("--image_dir",default='./data/IJCAI2019_data/twitter2017_images')

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

    parser.add_argument("--kernel_lst", default=[2,3,4], type=list, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")
    parser.add_argument("--char_lstm_hidden_size", default=64, type=int, help=" Char lstm hidden size")
    parser.add_argument("--before_att_size", default=256, type=int, help=" Char lstm hidden size")

    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--epochs", default=32, type=int, help="")

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
    label_vocab = {"[pad]": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7,"I-MISC": 8, "O": 9,"X":10}
    #word2vec
    word_embed,word_id = load_word_matrix(args)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data, dev_data, test_data,words = examples_convert_BERT(args,label_vocab,word_id)
    # s_preds, s_l,l_preds,l_labels = train_anttention_CRF(args,train_data, dev_data, test_data,word_embed,label_vocab)
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
    # with open('./wci_preds.txt','w',encoding='utf-8') as f:
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
    # with open('./wci_labels.txt','w',encoding='utf-8') as f:
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

    n_preds,labels,s_preds,preds_3,labels_3 = [],[],[],[],[]
    with open('./wc_preds.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            i = i.strip()
            pred = i.split(' ')
            pred = [int(j) for j in pred]
            n_preds.append(pred)
    with open('./wci_preds.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            i = i.strip()
            pred = i.split(' ')
            pred = [int(j) for j in pred]
            s_preds.append(pred)
    with open('./wc_labels.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            i = i.strip()
            pred = i.split(' ')
            pred = [int(j) for j in pred]
            labels.append(pred)

    id_label = {i: j for j, i in label_vocab.items()}
    evaluate_bdui2(n_preds, s_preds, labels, words, label_vocab, id_label)