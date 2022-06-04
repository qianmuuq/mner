import argparse
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import resnet
from mner_1 import LSTM_char, img_process
from ner_evaluate import evaluate
from only_text import load_word_matrix, set_seed, preprocess_word
from torchcrf import CRF
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_examples_BERT(args, file, chars):
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
                if len(sentence[1]) > 0:
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
    return sentences, chars, imgs, words

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
    word_mask = torch.zeros(sen_len).long()
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
        word_mask[i] = len(sen[0])-1
        for j, word in enumerate(sen[0]):
            word_input[i][j] = word_id.get(word, 1)
            mask_input[i][j] = 1
            char_mask[i][j] = len(word)-1
            for k, char in enumerate(word):
                #lstm,cnn char输入都是一样的
                char_input[i][j][k] = char_id.get(char, 1)
        for j, label in enumerate(sen[1]):
            labels[i][j] = label_vocab.get(label)
    return TensorDataset(word_input,char_input,char_mask,mask_input,labels,img_input,word_mask)


def examples_convert_BERT(args,label_vocab,word_id):
    chars = []
    train_s,chars,train_imgs,_ = get_examples_BERT(args,args.train_file,chars)
    dev_s,chars,dev_imgs,_ = get_examples_BERT(args,args.dev_file,chars)
    test_s,chars,test_imgs,test_words = get_examples_BERT(args,args.test_file,chars)

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

    return train_data,dev_data,test_data,test_imgs,test_words


class W_C_I_anttention_Vr_Gate(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_I_anttention_Vr_Gate, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_embed = image_embed
        self.vc = nn.Linear(2048,args.before_att_size)
        self.lstm_query = nn.LSTM(input_size=args.word_emb_dim+args.before_att_size,hidden_size=args.query_hidden_dim,bidirectional=False, batch_first=True)
        self.Pv = nn.Sequential(nn.Linear(args.before_att_size,args.before_att_size,bias=False),nn.Tanh())
        self.Pt = nn.Sequential(nn.Linear(args.query_hidden_dim,args.before_att_size,bias=False),nn.Tanh())
        self.Wa = nn.Linear(args.before_att_size,args.before_att_size)
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size=args.word_emb_dim+args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.Wv = nn.Linear(args.before_att_size,args.before_att_size)
        self.Uv = nn.Linear(args.before_att_size, args.before_att_size)
        self.Ww = nn.Linear(args.before_att_size, args.before_att_size)
        self.Uw = nn.Linear(args.before_att_size, args.before_att_size)
        self.sigmoid = nn.Sigmoid()
        self.Wm = nn.Linear(args.before_att_size, args.before_att_size)
        self.Um = nn.Linear(args.before_att_size, args.before_att_size)
        self.tanh = nn.Tanh()
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask,word_mask):
        # b,2048,7,7    b,2048
        img_areas,_ = self.image_embed(img_id)
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)

        b,s,_ = word_embed.size()

        #visual attention model
        word_char_embed = torch.cat([word_embed,char_embed],dim=-1)
        query,_ = self.lstm_query(word_char_embed,None)
        word_mask = word_mask.view(b,1,1).expand(b,1,query.size()[-1])
        query = query.gather(1,word_mask).view(b,-1)
        # b,49,2048
        img_areas = img_areas.view(b,2048,-1).permute(0,2,1)
        img_areas = self.vc(img_areas)
        Pt = self.Pt(query)
        Pv = self.Pv(img_areas)
        Pt = Pt.view(b,1,-1).expand(b,49,-1)
        A = Pv+Pt
        # b,2048,49
        Wa = self.Wa(A).permute(0,2,1)
        E = self.softmax(Wa)
        # b,512,49
        img_areas = img_areas.permute(0,2,1)
        vc = torch.mul(E,img_areas)
        # b,512
        vc = torch.sum(vc,dim=-1)

        #word char lstm
        vc_init = vc.view(b,2,-1).permute(1,0,2).contiguous()
        hc = torch.zeros_like(vc_init).cuda()
        w_c_lstm,_ = self.lstm(word_char_embed,(vc_init,hc))

        #visaul modulation gate
        vc = vc.view(b,1,-1)
        Wv = self.Wv(w_c_lstm)
        Uv = self.Uv(vc)
        Pv = self.sigmoid(Wv+Uv)
        Ww = self.Ww(w_c_lstm)
        Uw = self.Uw(vc)
        Pw = self.sigmoid(Ww + Uw)
        Wm = self.Wm(w_c_lstm)
        Um = self.Um(vc)
        m = self.tanh(Wm + Um)
        wm = torch.mul(Pw,w_c_lstm)+torch.mul(Pv,m)

        out = self.out(wm)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

class W_C_I_global_Vg(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_I_global_Vg, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_embed = image_embed
        self.vc = nn.Linear(2048,args.before_att_size)
        self.lstm = nn.LSTM(input_size=args.word_emb_dim+args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask,word_mask):
        # b,2048,7,7    b,2048
        _,img_global = self.image_embed(img_id)
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)

        b,s,_ = word_embed.size()

        word_char_embed = torch.cat([word_embed,char_embed],dim=-1)
        vc_init = self.vc(img_global).view(b,2,-1).permute(1,0,2).contiguous()
        hc = torch.zeros_like(vc_init).cuda()
        w_c_lstm,_ = self.lstm(word_char_embed,(vc_init,hc))

        out = self.out(w_c_lstm)
        return out

    def loss_fn(self, out, tags,masks):
        loss = self.crf(out,tags,masks,reduction='token_mean')
        return loss*-1

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.decode(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        return predicts

class W_C_I_Vc(nn.Module):
    def __init__(self,args,word_embed,char_embed,image_embed,labels_num=11):
        super(W_C_I_Vc, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed)
        self.char_embed = char_embed
        self.image_embed = image_embed
        self.vc = nn.Linear(2048,args.before_att_size)
        self.lstm_query = nn.LSTM(input_size=args.word_emb_dim+args.before_att_size,hidden_size=args.query_hidden_dim,bidirectional=False, batch_first=True)
        self.Pv = nn.Sequential(nn.Linear(args.before_att_size,args.before_att_size,bias=False),nn.Tanh())
        self.Pt = nn.Sequential(nn.Linear(args.query_hidden_dim,args.before_att_size,bias=False),nn.Tanh())
        self.Wa = nn.Linear(args.before_att_size,args.before_att_size)
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size=args.word_emb_dim+args.before_att_size, hidden_size=args.before_att_size//2,bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(args.before_att_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, labels_num)
        )
        self.crf = CRF(labels_num, batch_first=True)

    def forward(self,word_id,char_id,img_id,char_mask,word_mask):
        # b,2048,7,7    b,2048
        img_areas,_ = self.image_embed(img_id)
        # b,s,256
        char_embed = self.char_embed(char_id,char_mask)
        # b,s,200
        word_embed = self.word_embed(word_id)

        b,s,_ = word_embed.size()

        #visual attention model
        word_char_embed = torch.cat([word_embed,char_embed],dim=-1)
        query,_ = self.lstm_query(word_char_embed,None)
        word_mask = word_mask.view(b,1,1).expand(b,1,query.size()[-1])
        query = query.gather(1,word_mask).view(b,-1)
        # b,49,2048
        img_areas = img_areas.view(b,2048,-1).permute(0,2,1)
        img_areas = self.vc(img_areas)
        Pt = self.Pt(query)
        Pv = self.Pv(img_areas)
        Pt = Pt.view(b,1,-1).expand(b,49,-1)
        A = Pv+Pt
        # b,2048,49
        Wa = self.Wa(A).permute(0,2,1)
        E = self.softmax(Wa)
        # b,512,49
        img_areas = img_areas.permute(0,2,1)
        vc = torch.mul(E,img_areas)
        # b,512
        vc = torch.sum(vc,dim=-1)

        #word char lstm
        vc_init = vc.view(b,2,-1).permute(1,0,2).contiguous()
        hc = torch.zeros_like(vc_init).cuda()
        w_c_lstm,_ = self.lstm(word_char_embed,(vc_init,hc))

        out = self.out(w_c_lstm)

        #img_attention
        img_attention = torch.mean(E,dim=1).view(b,-1)
        i_mean = torch.mean(img_attention,dim=-1).view(b,1).expand(b,49)
        i_var = torch.std(img_attention,dim=-1).view(b,1)
        img_attention = (img_attention-i_mean)/i_var*-30
        img_attention = img_attention.view(b,7,7)
        return out,img_attention

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
    # resnet50 = resnet.resnet50()
    # resnet50.load_state_dict(torch.load(os.path.join(args.resnet50_dir,'resnet50-19c8e357.pth')), strict=True)
    # my_resnet = Myresnet(resnet50)
    #152
    resnet152 = resnet.resnet152()
    resnet152.load_state_dict(torch.load(os.path.join(args.resnet152_dir, 'resnet152-b121ed2d.pth')), strict=True)
    my_resnet = resnet.Myresnet_areas(resnet152)

    # model = W_C_I_global_Vg(args, word_embed, char_embed, my_resnet)
    model = W_C_I_Vc(args, word_embed, char_embed, my_resnet)
    # model = W_C_I_anttention_Vr_Gate(args,word_embed,char_embed,my_resnet)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00001)
    model.train()
    max_dev_f1, test_f1 = 0, 0
    for epoch in range(args.epochs):
        for batch in train_iter:
            word_input,char_input,char_mask,train_mask_t,train_labels_t,img_input,word_mask = batch
            word_input,char_input,char_mask,train_mask_t,train_labels_t,img_input,word_mask = word_input.cuda(),char_input.cuda(),char_mask.cuda(),train_mask_t.cuda(),train_labels_t.cuda(),img_input.cuda(),word_mask.cuda()

            out,_ = model(word_input,char_input,img_input,char_mask,word_mask)
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
                    word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input, word_mask = batch
                    word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input, word_mask = word_input.cuda(), char_input.cuda(), char_mask.cuda(), train_mask_t.cuda(), train_labels_t.cuda(), img_input.cuda(), word_mask.cuda()

                    out,_ = model(word_input, char_input, img_input, char_mask, word_mask)
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
                    img_a = None
                    for batch in test_iter:
                        word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input, word_mask = batch
                        word_input, char_input, char_mask, train_mask_t, train_labels_t, img_input, word_mask = word_input.cuda(), char_input.cuda(), char_mask.cuda(), train_mask_t.cuda(), train_labels_t.cuda(), img_input.cuda(), word_mask.cuda()

                        out,img_attention = model(word_input, char_input, img_input, char_mask, word_mask)
                        pred = model.predict(out, train_mask_t)
                        if img_a == None:
                            img_a = img_attention
                        else:
                            img_a = torch.cat([img_a,img_attention],dim=0)
                        for i, m in enumerate(train_labels_t.long()):
                            t1 = []
                            t2 = []
                            for j, _ in enumerate(train_labels_t[i]):
                                if train_labels_t[i][j].item() not in [0, 10]:
                                    t1.append(pred[i][j])
                                    t2.append(train_labels_t[i][j].item())
                            preds.append(t1)
                            labels.append(t2)
                    wordss = torch.zeros((len(preds), 50)).long()
                    acc, f1, p, r = evaluate(preds, labels, wordss, label_vocab)
                    print("test准确率:",acc,"F1:",f1)
            model.train()
    return img_a.cpu()

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
    parser.add_argument("--query_hidden_dim", default=512, type=int,help="Dimension of BiLSTM output, att layer (denoted as k) etc.")

    parser.add_argument("--kernel_lst", default=[2,3,4], type=list, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")
    parser.add_argument("--char_lstm_hidden_size", default=64, type=int, help=" Char lstm hidden size")
    parser.add_argument("--before_att_size", default=512, type=int, help=" Char lstm hidden size")

    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--epochs", default=25, type=int, help="")

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
    train_data, dev_data, test_data,test_img,test_words = examples_convert_BERT(args,label_vocab,word_id)
    img_a = train_anttention_CRF(args,train_data, dev_data, test_data,word_embed,label_vocab)

    for i,j in enumerate(test_img):
        try:
            image_path = os.path.join(args.image_dir, test_img[i])
            image = Image.open(image_path).convert('RGB')

            plt.imshow(image.resize((224, 224)))
            da = np.zeros((224, 224))
            for m in range(7):
                for n in range(7):
                    da[m*32:m*32+32,n*32:n*32+32] = img_a[i][m][n]
            plt.imshow(da, alpha=0.8, cmap='Greys')
            plt.title(' '.join(test_words[i]))
            plt.grid()
            plt.show()
        except:
            pass