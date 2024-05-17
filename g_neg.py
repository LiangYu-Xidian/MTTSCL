import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import configs
from utils import evaluate_func_1, evaluate_func_2
import models
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import pickle
import time

from keras.preprocessing.text import Tokenizer  #Using Theano backend
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

def label_encode(labels, classes):
    # classes = config.classes
    classes_dict = {c: i for i, c in enumerate(classes)}
    classes_dict["no_label"] = len(classes)
    print(classes_dict)
    labels_encode = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_encode

def read_full_data():
    data_df = pd.read_csv("./data/g_neg.csv")


    tok = Tokenizer(lower=False, char_level=True)  # 初始化标注器
    # tok.fit_on_texts(data[:10000]["seq"].values)  # 学习出文本的字典

    tok.word_index = {'L': 1, 'A': 2, 'V': 3, 'G': 4, 'E': 5, 'I': 6, 'S': 7, 'T': 8, 'D': 9, 'K': 10, \
        'R': 11, 'P': 12, 'F': 13, 'C': 20, 'X': 21, 'U': 22, 'B': 23, 'J': 24, 'Z': 25}

    print(tok.word_index)

    seqs = tok.texts_to_sequences(data_df["seq"].values)

    x = np.array(pad_sequences(seqs, config.max_len))

    # y1 = label_encode(data_df["label"].values.tolist(), ["C","CM","E","CW"])
    y = label_encode(data_df["label"].values.tolist(), ["C","CM","OM","E", "P"])
    
    
    return x, y


class MyDataSet(Data.Dataset):
    def __init__(self, x_data, y_label):
        super(MyDataSet, self).__init__()
        self.x = torch.LongTensor(x_data)
        self.y = torch.LongTensor(y_label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_func(sub_train_):
    model.train()
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=config.batch_size, shuffle=True)
    pbar = tqdm(data)
    for i, (batch_x,batch_y) in enumerate(pbar):
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
        output = model(batch_x)

        loss = criterion(output, batch_y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        train_acc += (output.argmax(1) == batch_y).sum().item()

        # pbar.set_description("batch {}".format(i))
        pbar.set_postfix(loss=loss.item())

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    model.eval()
    test_loss = 0
    test_acc = 0
    preds = []
    labels = []
    pp = []
    data = DataLoader(data_, batch_size=config.batch_size, shuffle=True)
    for i, (batch_x,batch_y) in enumerate(tqdm(data)):
        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
        with torch.no_grad():
            output = model(batch_x)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == batch_y).sum().item()

            preds += output.argmax(1).tolist()
            labels += batch_y.tolist()
            pp += output[:,1].tolist()
            # print(output.shape)

    print("Task g_neg")
    evaluate_func_2(labels,preds)

    return test_loss / len(data_), test_acc / len(data_)


if __name__ == "__main__":
    config = configs.config

    x, y= read_full_data()

    # weights1 = torch.tensor([1822,403,312,115], dtype=torch.float32)
    # weights1 = torch.tensor([max(weights1)/x for x in weights1])
    # # weights1 = torch.cat([weights1,torch.tensor([0])],0)
    # print(weights1)
    # # weight1 = weights1.to(config.device)
    # # print(config.device)
    # # print(weights1.device)

    weights2 = torch.tensor([5025,1655,541,509, 498], dtype=torch.float32)
    weights2 = torch.tensor([max(weights2)/x for x in weights2])
    # weights2 = torch.cat([weights2,torch.tensor([0])],0)
    print(weights2)
    # weights2 = weights2.to(config.device)

    # criterion1 = nn.CrossEntropyLoss(weight=weights1, ignore_index=len(config.classes_g_pos)).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=weights2, ignore_index=len(config.classes_g_neg)).to(config.device)

    kf = StratifiedKFold(n_splits = 5, random_state=2022, shuffle=False)

    acc_five_fold = 0

    for i, (train_idx, test_idx) in enumerate(kf.split(x,y)):
        print("train_index: {} , test_index: {}".format(len(train_idx), len(test_idx)))

        model = models.SingleTask_g_neg(config)
        model.to(config.device)
        # print(model)
        print("========================================")
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        train_set = MyDataSet(x[train_idx], y[train_idx])
        test_set = MyDataSet(x[test_idx], y[test_idx])

        for epoch in range(config.N_EPOCHS):
            print("------------------------fold {}----train epoch {}------------------------------".format(i, epoch+1))
            start_time = time.time()
            loss, acc = train_func(train_set)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print("loss: {}".format(loss))
            print("acc: {}".format(acc))

            # 查看测试集
            print('\033[92mChecking the results of test dataset...\33[0m')  #92 绿色 93 黄色
            loss, acc = test(test_set)
            print("loss: {}".format(loss))
            print("acc: {}".format(acc))
            # print(f'\033[92m\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)\33[0m')

            if epoch == config.N_EPOCHS-1:

                acc_five_fold += acc /5

    print("ACC:\t%.4f"% acc_five_fold)
