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
    data_df = pd.read_csv("./data/g_neg_pos.csv")

    data_df["fold_idx"] = data_df["label_g_neg"] + data_df["label_g_pos"]

    tok = Tokenizer(lower=False, char_level=True)  # 初始化标注器
    # tok.fit_on_texts(data[:10000]["seq"].values)  # 学习出文本的字典

    tok.word_index = {'L': 1, 'A': 2, 'V': 3, 'G': 4, 'E': 5, 'I': 6, 'S': 7, 'T': 8, 'D': 9, 'K': 10, \
        'R': 11, 'P': 12, 'F': 13, 'C': 20, 'X': 21, 'U': 22, 'B': 23, 'J': 24, 'Z': 25}

    print(tok.word_index)

    seqs = tok.texts_to_sequences(data_df["seq"].values)

    x = np.array(pad_sequences(seqs, config.max_len))

    y1 = label_encode(data_df["label_g_pos"].values.tolist(), ["C","CM","E","CW"])
    y2 = label_encode(data_df["label_g_neg"].values.tolist(), ["C","CM","OM","E", "P"])
    
    
    return x, y1, y2, data_df["fold_idx"]

class MyDataSet(Data.Dataset):
    def __init__(self, x_data, y1_label, y2_label):
        super(MyDataSet, self).__init__()
        self.x = torch.LongTensor(x_data)
        self.y1 = torch.LongTensor(y1_label)
        self.y2 = torch.LongTensor(y2_label)
        self.y1_mask = torch.LongTensor(y1_label).lt(len(config.classes_g_pos))
        self.y2_mask = torch.LongTensor(y2_label).lt(len(config.classes_g_neg))
        # self.y2_mask = torch.LongTensor(y2_label.sum(axis=1)>0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y1_mask[idx], self.y2[idx], self.y2_mask[idx]

def train_func(sub_train_):
    model.train()
    # Train the model
    train_loss_1, train_loss_2, total_loss = 0, 0, 0
    train_acc_1, train_acc_2 = 0, 0
    y1_num, y2_num = 0, 0
    label_1, label_2, pred_1,pred_2 = [], [], [], []
    data = DataLoader(sub_train_, batch_size=config.batch_size, shuffle=True)
    pbar = tqdm(data)
    for i, (batch_x, batch_y1, mask_y1, batch_y2, mask_y2) in enumerate(pbar):
        optimizer.zero_grad()
        batch_x, batch_y1, batch_y2 = batch_x.to(config.device), batch_y1.to(
            config.device), batch_y2.to(config.device)
        output = model(batch_x)

        loss_1 = criterion1(output[0], batch_y1)
        loss_2 = criterion2(output[1], batch_y2)
        loss = 0.5 * loss_1 + loss_2

        train_loss_1 += loss_1.item()
        train_loss_2 += loss_2.item()

        loss.backward()
        optimizer.step()

        y1_num += mask_y1.sum().item()
        y2_num += mask_y2.sum().item()

        # print(loss_1.item()/y1_num)
        # print(loss_2.item()/y2_num)

        train_acc_1 += (output[0].argmax(1) == batch_y1).sum().item()
        train_acc_2 += (output[1].argmax(1) == batch_y2).sum().item()

        # print(train_acc_1/y1_num)
        # print(train_acc_2/y2_num)

        p1 = output[0].argmax(1)
        p2 = output[1].argmax(1)

        label_1 += batch_y1[np.where(mask_y1)].tolist()
        pred_1 += p1[np.where(mask_y1)].tolist()
        label_2 += batch_y2[np.where(mask_y2)].tolist()
        pred_2 += p2[np.where(mask_y2)].tolist()

        pbar.set_description("total_loss {:.4f}".format(loss.item()))
        pbar.set_postfix(loss_g=loss_1.item(),loss_a = loss_2.item())

    print("task_1 : g_pos")
    evaluate_func_1(label_1,pred_1)
    print("task_2 : g_neg")
    evaluate_func_2(label_2,pred_2)

    return train_loss_1 / y1_num, train_loss_2 / y2_num, train_acc_1 / y1_num, train_acc_2 / y2_num


def test(data_):
    model.eval()
    test_loss_1, test_loss_2, total_loss = 0, 0, 0
    test_acc_1, test_acc_2 = 0, 0
    y1_num, y2_num = 0, 0
    label_1, label_2, pred_1,pred_2 = [], [], [], []
    data = DataLoader(data_, batch_size=config.batch_size, shuffle=True)
    pbar = tqdm(data)
    for i, (batch_x, batch_y1, mask_y1, batch_y2, mask_y2) in enumerate(pbar):
        batch_x, batch_y1, batch_y2 = batch_x.to(config.device), batch_y1.to(
            config.device), batch_y2.to(config.device)
        with torch.no_grad():
            output = model(batch_x)
            loss_1 = criterion1(output[0], batch_y1)
            loss_2 = criterion2(output[1], batch_y2)
            loss = 0.5 * loss_1 + loss_2

            y1_num += mask_y1.sum().item()
            y2_num += mask_y2.sum().item()
            total_loss += loss.item()
            test_loss_1 += loss_1.item()
            test_loss_2 +=loss_2.item()

            p1 = output[0].argmax(1)
            p2 = output[1].argmax(1)

            test_acc_1 += (p1 == batch_y1).sum().item()
            test_acc_2 += (p2 == batch_y2).sum().item()

            label_1 += batch_y1[np.where(mask_y1)].tolist()
            pred_1 += p1[np.where(mask_y1)].tolist()
            label_2 += batch_y2[np.where(mask_y2)].tolist()
            pred_2 += p2[np.where(mask_y2)].tolist()

            pbar.set_description("total_loss {:.4f}".format(loss.item()))
            pbar.set_postfix(loss_g=loss_1.item(),loss_a = loss_2.item())
    
    print("task_1 : g_pos")
    evaluate_func_1(label_1,pred_1)
    print("task_2 : g_neg")
    evaluate_func_2(label_2,pred_2)

    return test_loss_1 / y1_num, test_loss_2 / y2_num, test_acc_1 / y1_num, test_acc_2 / y2_num

if __name__ == "__main__":
    config = configs.config

    x, y1, y2, fold_idx = read_full_data()

    weights1 = torch.tensor([1822,403,312,115], dtype=torch.float32)
    weights1 = torch.tensor([max(weights1)/x for x in weights1])
    # weights1 = torch.cat([weights1,torch.tensor([0])],0)
    print(weights1)
    # weight1 = weights1.to(config.device)
    # print(config.device)
    # print(weights1.device)

    weights2 = torch.tensor([5025,1655,541,509, 498], dtype=torch.float32)
    weights2 = torch.tensor([max(weights2)/x for x in weights2])
    # weights2 = torch.cat([weights2,torch.tensor([0])],0)
    print(weights2)
    # weights2 = weights2.to(config.device)

    criterion1 = nn.CrossEntropyLoss(weight=weights1, ignore_index=len(config.classes_g_pos)).to(config.device)
    criterion2 = nn.CrossEntropyLoss(weight=weights2, ignore_index=len(config.classes_g_neg)).to(config.device)


    kf = StratifiedKFold(n_splits = 5, random_state=2022, shuffle=False)

    fold_dict = {}

    acc_1_five_fold , acc_2_five_fold = 0,0

    for i, (train_idx, test_idx) in enumerate(kf.split(x,fold_idx)):
        print("train_index: {} , test_index: {}".format(len(train_idx), len(test_idx)))

        model = models.CustomizedShare(config)
        model.to(config.device)
        print("========================================")
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        train_set = MyDataSet(x[train_idx], y1[train_idx], y2[train_idx])
        test_set = MyDataSet(x[test_idx], y1[test_idx], y2[test_idx])

        for epoch in range(config.N_EPOCHS):
            print("------------------------fold {}----train epoch {}------------------------------".format(i, epoch+1))
            start_time = time.time()
            loss_1, loss_2, acc_1, acc_2 = train_func(train_set)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print("loss_1: {}".format(loss_1))
            print("loss_2: {}".format(loss_2))
            print("acc_1: {}".format(acc_1))
            print("acc_2: {}".format(acc_2))

            # 查看测试集
            print('\033[92mChecking the results of test dataset...\33[0m')  #92 绿色 93 黄色
            loss_1, loss_2, acc_1, acc_2 = test(test_set)
            print("test loss_1: {}".format(loss_1))
            print("test loss_2: {}".format(loss_2))
            print("test acc_1: {}".format(acc_1))
            print("test acc_2: {}".format(acc_2))
            # print(f'\033[92m\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)\33[0m')

            if epoch == config.N_EPOCHS-1:

                acc_1_five_fold += acc_1 /5
                acc_2_five_fold += acc_2 /5

    print("ACC_g_pos:\t%.4f"% acc_1_five_fold)
    print("ACC_g_neg:\t%.4f"% acc_2_five_fold)