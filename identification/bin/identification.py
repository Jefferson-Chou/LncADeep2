from .embedding import *
from Bio import SeqIO
from itertools import repeat
import shutil 
import numpy as np
import pandas as pd
import torch
import pickle
from torch import nn, Tensor
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import configparser

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
        return_dict = {}
        return_dict['logits'] = logits
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
config.num_train_epochs = 50
config.print_every_epoch = 5
config.output_mode = "boolean_cls"

model = MLP_learner(config)


def identify(filename, device = 'cpu', out_dir = './identification/output/results.csv', hmmsearch_thread = 1, feat_out = 0, model_file = './identification/models/model_93_0.1750_0.9440.pt'):
    device = torch.device(device)
    # parse input file
    print('reading fasta files...')
    seq = SeqIO.parse(filename, 'fasta')
    print('calculating features...')
    SeqID, SeqList, HMMdict1, logscore_dict = pre(filename, seq, hmmsearch_thread)
    para_tuple_list = zip(SeqID, SeqList, repeat(HMMdict1), repeat(logscore_dict))
    emb = IntactFeature(para_tuple_list)

    # scaling
    with open('./identification/models/training_sets.pkl', 'rb') as f:
        emb1, emb2 = pickle.load(f)
    emb1, emb2 = torch.tensor(emb1, dtype=torch.float32), torch.tensor(emb2, dtype=torch.float32)
    y = np.array([1] * emb1.shape[0] + [0] * emb2.shape[0])
    X = torch.vstack((emb1, emb2))
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(y)
    X_train_mean = torch.mean(X_train, dim=0)
    X_train_sd = torch.std(X_train, dim=0)
    emb = torch.tensor(emb, dtype=torch.float32)
    emb = (emb - X_train_mean) / X_train_sd
    emb = emb.to(device)

    if feat_out == 1:
        emb_dir = f'{os.path.dirname(out_dir)}/emb.pkl'
        with open(emb_dir, 'wb') as f:
            pickle.dump(emb, f)
        print(f'features are saved in {emb_dir}')

    # model prediction
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()
    print('predicting...')
    output = model(feature = emb)
    logits = output['logits'].detach().cpu().numpy()
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(torch.tensor(logits, dtype=torch.float32))[:,1].numpy()
    preds = np.argmax(logits, axis=1)

    # output
    results = pd.DataFrame({'SeqID': SeqID, 'Coding_probability': [1 - prob for prob in probs], 'Prediction': ['Non-coding' if i == 1 else 'Coding' for i in preds]})
    results.to_csv(out_dir, index=False, quoting=False)
    shutil.rmtree('./identification/tmp/')  
    os.mkdir('./identification/tmp/') 
    print(f'finished! results are saved in {out_dir}')

