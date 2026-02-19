import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric import transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

''' Data preparation '''
# interaction data
lpi = pd.read_csv('./annotation/src/lpi_filt.csv', sep = '\t')
ppi = pd.read_csv('./annotation/src/ppi_filt.csv', sep = '\t')
pmi = pd.read_csv('./annotation/src/pmi_filt.csv', sep = '\t')
lmi = pd.read_csv('./annotation/src/lmi_filt.csv', sep = '\t')

with open('./annotation/src/ncID2fea_bert.pkl', 'rb') as f:
    ncid2fea = pickle.load(f)
with open('./annotation/src/miID2fea_bert.pkl', 'rb') as f:
    miid2fea = pickle.load(f)
tarname2fea = pd.read_csv('./annotation/src/tarName2fea_protrans.csv')
del tarname2fea['Unnamed: 0']

miID = list(dict.fromkeys(miid2fea['miID']))
miID_dict = {key: value for value, key in enumerate(miID)}
ncID = list(dict.fromkeys(ncid2fea['ncID']))
ncID_dict = {key: value for value, key in enumerate(ncID)}
tarName = list(dict.fromkeys(tarname2fea['tarName']))
tarName_dict = {key: value for value, key in enumerate(tarName)}

lpi_idx = lpi.copy()
lpi_idx['ncIDX'] = lpi['geneID'].map(ncID_dict)
lpi_idx['tarIDX'] = lpi['RBP'].map(tarName_dict)
lpi_idx = lpi_idx.iloc[:,2:]
lmi_idx = lmi.copy()
lmi_idx['ncIDX'] = lmi['geneID'].map(ncID_dict)
lmi_idx['miIDX'] = lmi['miRNAid'].map(miID_dict)
lmi_idx = lmi_idx.iloc[:,2:]
pmi_idx = pmi.copy()
pmi_idx['tarIDX'] = pmi['RBP'].map(tarName_dict)
pmi_idx['miIDX'] = pmi['miRNA'].map(miID_dict)
pmi_idx = pmi_idx.iloc[:,2:]
ppi_idx = ppi.copy()
ppi_idx['tarIDXa'] = ppi['protein1'].map(tarName_dict)
ppi_idx['tarIDXb'] = ppi['protein2'].map(tarName_dict)
ppi_idx = ppi_idx.iloc[:,2:]

with open("./annotation/src/edge_label_index_2hop.pt", "rb") as f:
    edge_label_index_2hop = pickle.load(f)

# molecular features
pro_fea = tarname2fea.iloc[:,1:]
pro_fea = pro_fea.astype(float)
pro_fea = torch.from_numpy(pro_fea.values)
lncrna_fea = ncid2fea.iloc[:,1:]
lncrna_fea = lncrna_fea.astype(float)
lncrna_fea = torch.from_numpy(lncrna_fea.values)
mirna_fea = miid2fea.iloc[:,1:]
mirna_fea = mirna_fea.astype(float)
mirna_fea = torch.from_numpy(mirna_fea.values)
pro_fea_mean = torch.mean(pro_fea, dim=0)
pro_fea_sd = torch.std(pro_fea, dim=0)
pro_fea = (pro_fea - pro_fea_mean) / pro_fea_sd
lncrna_fea_mean = torch.mean(lncrna_fea, dim=0)
lncrna_fea_sd = torch.std(lncrna_fea, dim=0)
lncrna_fea = (lncrna_fea - lncrna_fea_mean) / lncrna_fea_sd
mirna_fea_mean = torch.mean(mirna_fea, dim=0)
mirna_fea_sd = torch.std(mirna_fea, dim=0)
mirna_fea = (mirna_fea - mirna_fea_mean) / mirna_fea_sd

# heterogenous graph construction
data = HeteroData()
data['lncrna'].node_id = torch.arange(len(list(ncID_dict.values())))
data['protein'].node_id = torch.arange(len(list(tarName_dict.values())))
data['mirna'].node_id = torch.arange(len(list(miID_dict.values())))
data['lncrna'].x = lncrna_fea
data['protein'].x = pro_fea
data['mirna'].x = mirna_fea

ncID_idx = torch.from_numpy(lpi_idx['ncIDX'].values)
tarName_idx = torch.from_numpy(lpi_idx['tarIDX'].values)
edge_idx_lpi = torch.stack([ncID_idx, tarName_idx], dim=0)
edge_idx_lpi = edge_idx_lpi.to(torch.long)
data['lncrna','lpi','protein'].edge_index = edge_idx_lpi

ncID_idx = torch.from_numpy(lmi_idx['ncIDX'].values.astype(float))
miID_idx = torch.from_numpy(lmi_idx['miIDX'].values.astype(float))
edge_idx_lmi = torch.stack([ncID_idx, miID_idx], dim=0)
edge_idx_lmi = edge_idx_lmi.to(torch.long)
data['lncrna','lmi','mirna'].edge_index = edge_idx_lmi

tarName_idx = torch.from_numpy(pmi_idx['tarIDX'].values.astype(float))
miID_idx = torch.from_numpy(pmi_idx['miIDX'].values.astype(float))
edge_idx_pmi = torch.stack([tarName_idx, miID_idx], dim=0)
edge_idx_pmi = edge_idx_pmi.to(torch.long)
data['protein','pmi','mirna'].edge_index = edge_idx_pmi

data = T.ToUndirected()(data) 

tarName_idxA = torch.from_numpy(ppi_idx['tarIDXa'].values.astype(float))
tarName_idxB = torch.from_numpy(ppi_idx['tarIDXb'].values.astype(float))
edge_idx_ppi = torch.stack([tarName_idxA, tarName_idxB], dim=0)
edge_idx_ppi = edge_idx_ppi.to(torch.long)
data['protein','ppi','protein'].edge_index = edge_idx_ppi

''' GNN Model '''
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        return x
    
    
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lncrna_ln1 = nn.Linear(768, 300)
        self.lncrna_ln2 = nn.Linear(300, 200)
        self.lncrna_ln3 = nn.Linear(200, 150)
        self.hidden1 = nn.Linear(300, 150)
        self.hidden2 = nn.Linear(150, 50)
        self.hidden3 = nn.Linear(50, 3)

    def forward(self, x_lncrna: Tensor, x_protein: Tensor, edge_label_index: Tensor) -> Tensor:
        x_lncrna = x_lncrna.to(torch.float32)
        x_lncrna = F.relu(self.lncrna_ln1(x_lncrna))
        x_lncrna = F.relu(self.lncrna_ln2(x_lncrna))
        x_lncrna = self.lncrna_ln3(x_lncrna)
        
        edge_feat_lncrna = x_lncrna[edge_label_index[0]] 
        edge_feat_protein = x_protein[edge_label_index[1]]
        edge_feat = torch.hstack((edge_feat_lncrna, edge_feat_protein))
        edge_feat = F.relu(self.hidden1(edge_feat))
        edge_feat = F.relu(self.hidden2(edge_feat))
        edge_feat = self.hidden3(edge_feat)
        return edge_feat
        
    
    def forward_class(self, x_lncrna: Tensor, x_protein: Tensor, edge_label_index: Tensor) -> Tensor:
        x_lncrna = x_lncrna.to(torch.float32)
        x_lncrna = F.relu(self.lncrna_ln1(x_lncrna))
        x_lncrna = F.relu(self.lncrna_ln2(x_lncrna))
        x_lncrna = self.lncrna_ln3(x_lncrna)
        
        edge_feat_lncrna = x_lncrna[edge_label_index[0]]
        edge_feat_protein = x_protein[edge_label_index[1]]
        edge_feat = torch.hstack((edge_feat_lncrna, edge_feat_protein))
        return edge_feat

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lncrna_emb = torch.nn.Linear(768, hidden_channels)
        self.mirna_emb = torch.nn.Linear(768, hidden_channels)
        self.protein_emb = torch.nn.Linear(1024, hidden_channels)
        self.gnn = GNN(hidden_channels)
        
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
              
        x_dict = {
            "lncrna": self.lncrna_emb(data["lncrna"].x.to(torch.float)),
            "mirna": self.mirna_emb(data["mirna"].x.to(torch.float)),
            "protein": self.protein_emb(data["protein"].x.to(torch.float))
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            data["lncrna"].x,
            x_dict["protein"],
            data["lncrna", "lpi", "protein"].edge_label_index
        )
        return pred
    
    def forward_graph(self, data: HeteroData) -> Tensor:    
        x_dict = {
            "lncrna": self.lncrna_emb(data["lncrna"].x.to(torch.float)),
            "mirna": self.mirna_emb(data["mirna"].x.to(torch.float)),
            "protein": self.protein_emb(data["protein"].x.to(torch.float))
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict

    def forward_classifer(self, data: HeteroData) -> Tensor:
        x_dict = {
            "lncrna": self.lncrna_emb(data["lncrna"].x.to(torch.float)),
            "mirna": self.mirna_emb(data["mirna"].x.to(torch.float)),
            "protein": self.protein_emb(data["protein"].x.to(torch.float))
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier.forward_class(
            data["lncrna"].x,
            x_dict["protein"],
            data["lncrna", "lpi", "protein"].edge_label_index
        )
        return pred
    
    def forward_pre(self, data: HeteroData) -> Tensor:
        x = self.lncrna_emb(data["lncrna"].x.to(torch.float)),
        return x

gnn_mod = Model(hidden_channels=150)

''' Data preparation for embedding '''
lnc2uniprot_pos = pd.read_csv('./annotation/src/lnc2uniprot_bp_pos_old.csv')
lnc2uniprot_neg = pd.read_csv('./annotation/src/uniprot2lnc_bp_neg_old.csv')

lnc2uniprot_pos_idx = pd.DataFrame(columns=['lnc', 'uniprot'])
lnc2uniprot_pos_idx['lnc'] = lnc2uniprot_pos['lnc'].map(ncID_dict)
lnc2uniprot_pos_idx['uniprot'] = lnc2uniprot_pos['uniprot'].map(tarName_dict)

lnc2uniprot_neg_idx = pd.DataFrame(columns=['lnc', 'uniprot'])
lnc2uniprot_neg_idx['lnc'] = lnc2uniprot_neg['lnc'].map(ncID_dict)
lnc2uniprot_neg_idx['uniprot'] = lnc2uniprot_neg['uniprot'].map(tarName_dict)

lncrna_fea = ncid2fea.iloc[:,1:].astype(float)
lncrna_fea = torch.from_numpy(lncrna_fea.values)

lncrna_fea_mean = torch.mean(lncrna_fea, dim=0)
lncrna_fea_sd = torch.std(lncrna_fea, dim=0)

''' BERT embedding '''
from .bert_embedding import *

with open(f"./annotation/src/train_test_data_fold1.pt", "rb") as f:
    train_data, _ = pickle.load(f)

''' GNN embedding '''
# embedding for lncRNA of the interaction network
prolist_default = set(lnc2uniprot_pos_idx['uniprot'])

def export_emb_in(ensembl_id, device, prolist = prolist_default):
    ncid = ncID_dict[ensembl_id]
    # custom data
    gnn_mod.load_state_dict(torch.load('./annotation/models/lpi_model_50_0.9342.pt', map_location=device))
    gnn_mod.to(device)
    gnn_mod.eval()
    
    with torch.no_grad():
        edge_label_index = torch.tensor([[ncid] * len(prolist), 
                                        [uniprotid for uniprotid in prolist]], dtype=torch.long)
        
        datum = train_data.clone()
        datum["lncrna", "lpi", "protein"].edge_label_index = edge_label_index 
        datum = datum.to(device)
        custom_emb = gnn_mod.forward_classifer(datum)
        with open(f'./annotation/tmp/custom_emb_{ensembl_id}.pkl', 'wb') as f:
            pickle.dump((custom_emb, edge_label_index), f)
    
# embedding for lncRNA out of the interaction network
def export_emb_ex(ensembl_id, device, lnc_emb_df, prolist = prolist_default):
    ncid = 12650
    # custom data
    gnn_mod.load_state_dict(torch.load('./annotation/models/lpi_model_50_0.9342.pt', map_location=device))

    gnn_mod.to(device)
    gnn_mod.eval()
    
    with torch.no_grad():
        edge_label_index = torch.tensor([[ncid] * len(prolist), 
                                        [uniprotid for uniprotid in prolist]], dtype=torch.long)
        
        datum = train_data.clone()
        datum["lncrna", "lpi", "protein"].edge_label_index = edge_label_index 
        datum['lncrna']['node_id'] = torch.cat((datum['lncrna']['node_id'], torch.tensor([12650])))
        lnc_fea = torch.tensor(lnc_emb_df.loc[lnc_emb_df['ncID'] == ensembl_id].iloc[:,1:].values)
        
        lnc_fea = (lnc_fea - lncrna_fea_mean) / lncrna_fea_sd
        datum['lncrna'].x = torch.vstack((datum['lncrna'].x, lnc_fea))
        datum = datum.to(device)
        custom_emb = gnn_mod.forward_classifer(datum)
        with open(f'./annotation/tmp/custom_emb_{ensembl_id}.pkl', 'wb') as f:
            pickle.dump((custom_emb, edge_label_index), f)

def export_emb(filename, dev):
    lnc_emb_df = BERT_embedding(filename=filename, device=dev)
    lnc_emb_df.reset_index(drop=True, inplace=True)
    device = torch.device(dev)
    for id in lnc_emb_df['ncID']:
        if id in ncID:
            export_emb_in(id, device)
        else:
            export_emb_ex(id, device, lnc_emb_df)

def pred_lpi_in(ensembl_id, device, prolist):
    softmax = torch.nn.Softmax(dim=1)
    ncid = ncID_dict[ensembl_id]
    gnn_mod.load_state_dict(torch.load('./annotation/models/lpi_model_50_0.9342.pt', map_location=device))
    gnn_mod.to(device)
    gnn_mod.eval()
    lpi_res = pd.DataFrame(columns=['protein', 'interaction_type', 'probability'])
    with torch.no_grad():
        edge_label_index = torch.tensor([[ncid] * len(prolist), 
                                        [uniprotid for uniprotid in prolist]], dtype=torch.long)
        
        datum = train_data.clone()
        datum["lncrna", "lpi", "protein"].edge_label_index = edge_label_index 
        datum = datum.to(device)
        logit = gnn_mod.forward(datum)
        prob = softmax(logit)
        pred = np.argmax(prob.cpu().numpy(), axis=1)
        lpi_res = pd.DataFrame({'protein': [tarName[idx] for idx in prolist],
                                'interaction_type': pred,
                                'probability': np.max(prob.cpu().numpy(), axis=1)})
        return lpi_res

def pred_lpi_ex(ensembl_id, device, lnc_emb_df, prolist):
    softmax = torch.nn.Softmax(dim=1)
    ncid = 12650
    gnn_mod.load_state_dict(torch.load('./annotation/models/lpi_model_50_0.9342.pt', map_location=device))
    gnn_mod.to(device)
    gnn_mod.eval()
    
    with torch.no_grad():
        edge_label_index = torch.tensor([[ncid] * len(prolist), 
                                        [uniprotid for uniprotid in prolist]], dtype=torch.long)
        
        datum = train_data.clone()
        datum["lncrna", "lpi", "protein"].edge_label_index = edge_label_index 
        datum['lncrna']['node_id'] = torch.cat((datum['lncrna']['node_id'], torch.tensor([12650])))
        lnc_fea = torch.tensor(lnc_emb_df.loc[lnc_emb_df['ncID'] == ensembl_id].iloc[:,1:].values)
        
        lnc_fea = (lnc_fea - lncrna_fea_mean) / lncrna_fea_sd
        datum['lncrna'].x = torch.vstack((datum['lncrna'].x, lnc_fea))
        datum = datum.to(device)
        logit = gnn_mod.forward(datum)
        prob = softmax(logit)
        pred = np.argmax(prob.cpu().numpy(), axis=1)
        lpi_res = pd.DataFrame({'protein': [tarName[idx] for idx in prolist],
                                'interaction_type': pred,
                                'probability': np.max(prob.cpu().numpy(), axis=1)})
        return lpi_res

def pred_lpi(filename, dev, lpi_out = './annotation/output/'):
    prolist = list(tarName_dict.values())
    lnc_emb_df = BERT_embedding(filename=filename, device=dev)
    lnc_emb_df.reset_index(drop=True, inplace=True)
    device = torch.device(dev)
    for id in lnc_emb_df['ncID']:
        if id in ncID:
            lpi_res = pred_lpi_in(id, device, prolist)
        else:
            lpi_res = pred_lpi_ex(id, device, lnc_emb_df, prolist)
        lpi_res.to_csv(os.path.join(lpi_out, f'{id}_lpi_prediction.csv'), index=False)
    print('LPI prediction completed!')