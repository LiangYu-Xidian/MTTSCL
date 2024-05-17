from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
import numpy as np

def sen(Y_test,Y_pred,n): 
    # n is classfication nums
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def spe(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe

def ACC(Y_test,Y_pred,n):
    
    acc = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    print("===========================")
    print(con_mat)
    print("===========================")
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
        
    return acc

def pre(Y_test,Y_pred,n):
    
    pre = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)
        
    return pre


# def evaluate_func(targets, preds):
#     target_names = ["Cytoplasmic","Cytoplasmic Membrane","Extracellular","Cellwall"]
#     print(classification_report(targets, preds, target_names=target_names,digits=4))

def evaluate_func_1(targets, preds):
    target_names = ["Cytoplasmic","Cytoplasmic Membrane","Extracellular","Cellwall"]
    print(classification_report(targets, preds, target_names=target_names,digits=4))
    # sensitivity = sen(targets, preds, 4)
    # specificity = spe(targets, preds, 4)
    # acc = ACC(targets, preds, 4)
    # print("\t\t\t accuracy  sensitivity  specificity")
    # for i in range(4):
    #     print("{:>20}{:>12.4f}{:>12.4f}{:>12.4f}".format(target_names[i],acc[i],sensitivity[i],specificity[i]))


def evaluate_func_2(targets, preds):
    target_names = ["Cytoplasmic","Cytoplasmic Membrane","Outer Membrane","Extracellular","Periplasmic"]
    print(classification_report(targets, preds, target_names=target_names,digits=4))
