import numpy as np
import pandas as pd
import time
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, hamming_loss, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import configparser
from accelerate import Accelerator
from transformers import AdamW

from .bin.embedding import *
from itertools import repeat
from Bio import SeqIO

'''model'''
class Simple_Head(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cls_dropout_prob = config.dropout
        self.feature_size = config.feature_size
        
        self.in_proj = torch.nn.Linear(self.feature_size, self.hidden_size[0])
        self.hidden_1 = torch.nn.Linear(self.hidden_size[0] , self.hidden_size[1])
        self.hidden_2 = torch.nn.Linear(self.hidden_size[1] , self.hidden_size[2])
        self.out_proj = torch.nn.Linear(self.hidden_size[-1], self.num_labels)
        self.batchnorm0 = nn.BatchNorm1d(self.hidden_size[0])
        self.dropout = torch.nn.Dropout(self.cls_dropout_prob) 
    
    def forward(self, features, **kwargs):
        x = features
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.dropout(F.relu(self.batchnorm0(self.in_proj(x))))
        x = self.dropout(F.relu(self.hidden_1(x)))
        x = self.dropout(F.relu(self.hidden_2(x)))
        x = self.out_proj(x)
        return x 
    
class MLP_learner(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model_type = 'MLP_learner'
        self.dropout = config.dropout
        self.num_labels = config.num_labels
        self.classifier = Simple_Head(config)
        self.criterion = CrossEntropyLoss
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, feature=None, labels=None, weights=None):
        logits = self.classifier(feature)
        loss_fct = self.criterion()
        loss = loss_fct(logits, labels)

        return_dict = {}
        return_dict['logits'] = logits
        return_dict['loss'] = loss
        return return_dict

'''Configs'''
config = configparser.ConfigParser()
config.num_labels = 2
config.feature_size = 53
config.hidden_size = [128, 64, 32]
config.dropout = 0.1
config.BS = 256
config.learning_rate = 5e-5
config.weight_decay = 1e-2
config.num_train_epochs = 100
config.print_every_epoch = 5
config.output_mode = "boolean_cls"

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def metrics(preds, labels, probs):
    acc = simple_accuracy(preds, labels)
    precision_mi = precision_score(y_true=labels, y_pred=preds, average='micro')
    recall_mi = recall_score(y_true=labels, y_pred=preds, average='micro')
    precision_ma = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall_ma = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    hamming = hamming_loss(y_true=labels, y_pred=preds)
    auc_micro = roc_auc_score(y_true = labels, y_score = probs, average='macro', multi_class = 'ovo')
    auc_macro = roc_auc_score(y_true = labels, y_score = probs, average='macro', multi_class = 'ovo')
    
    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "hamming_loss":hamming,
        "precision_mi": precision_mi,
        "recall_mi": recall_mi,
        "precision_ma": precision_ma,
        "recall_ma": recall_ma,
        "auc_micro":auc_micro,
        "auc_macro":auc_macro,
    }

def binar_metrics(preds, labels, probs):
    acc = simple_accuracy(preds, labels)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    return {
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "cm": cm,
    }

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, train_dataloader, optimizer, accelerator):
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_dataloader):
        feature=batch[0]
        labels=batch[1]
        outputs = model(feature, labels)
        loss = outputs['loss']
        epoch_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()

    return epoch_loss/len(train_dataloader)


def evaluate(model, dataloader):
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    epoch_loss = 0
    logits = None
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            feature=batch[0]
            labels=batch[1]
            outputs = model(feature, labels)
        
        if logits is None:
            logits = outputs['logits'].detach().cpu().numpy()
            reference = labels.detach().cpu().numpy()
        else:
            logits = np.append(logits,outputs['logits'].detach().cpu().numpy(), axis=0)
            reference = np.append(reference,labels.detach().cpu().numpy(), axis=0)

        loss = outputs['loss']
        epoch_loss += loss.item()
    
    eval_loss = epoch_loss/len(dataloader)
    
    if config.output_mode == "boolean_cls":
        probs = softmax(torch.tensor(logits, dtype=torch.float32))[:,1].numpy()
        preds = np.argmax(logits, axis=1)
        results = binar_metrics(preds, reference, probs)
    return eval_loss, preds, results, reference, probs

def Train(filename, label_file, device = 'cpu', hmmsearch_thread = 1):
    device = torch.device(device)
    # parse input file
    print('reading fasta files...')
    seq = SeqIO.parse(filename, 'fasta')
    print('calculating features...')
    SeqID, SeqList, HMMdict1, logscore_dict = pre(filename, seq, hmmsearch_thread)
    para_tuple_list = zip(SeqID, SeqList, repeat(HMMdict1), repeat(logscore_dict))
    emb = IntactFeature(para_tuple_list)
    with open(label_file, 'r') as f:
        labels = f.readlines()
        labels = [int(line.strip()) for line in labels]
    
    y = np.array(labels)
    X_train = torch.Tensor(emb)
    y_train = torch.LongTensor(y)

    X_train_mean = torch.mean(X_train, dim=0)
    X_train_sd = torch.std(X_train, dim=0)
    X_train = (X_train - X_train_mean) / X_train_sd

    # Dataset                  
    train_set = TensorDataset(X_train.to(device), y_train.to(device))

    # weighted sampler
    class_count = pd.DataFrame(y_train).value_counts().sort_index().to_list()
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights[y_train],
        num_samples=X_train.shape[0],
        replacement=True
    )

    # Dataloader
    train_dataloader = DataLoader(train_set, batch_size=config.BS, sampler=weighted_sampler)
    ori_train_dataloader = DataLoader(train_set, batch_size=config.BS)

    '''model'''
    model = MLP_learner(config).to(device)

    no_decay = ["bias", "LayerNorm.weight"] 
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    accelerator = Accelerator() 
    
    model, optimizer, train_dataloader, ori_train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, ori_train_dataloader
        )
    
    # Training
    best_valid_loss = float('inf')
    best_epoch  = 0 
    for epoch in range(config.num_train_epochs):

        start_time = time.monotonic()

        train_loss = train(model, train_dataloader, optimizer, accelerator)
        _, y_pred, results, _, _ = evaluate(model, train_dataloader)
        
        auc = results['auc']
        rec = results['recall']
        prec = results['precision']
        cm = results['cm']
        f1 = results['f1']
        
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} ')
        print(f'\t Train. Auc: {auc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | F1: {f1:.4f} ')
        print(cm)
        torch.save(model.state_dict(), f'./identification/models/custom/model_{epoch}_{train_loss:.4f}_{f1:.4f}.pt')
    
    print(f'Training finished! Models saved in ./identification/models/custom/')


