from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import pickle
from Bio import SeqIO

with open('./annotation/src/ncID2fea_bert.pkl', 'rb') as f:
    ncid2fea = pickle.load(f)

ncID = list(dict.fromkeys(ncid2fea['ncID']))
ncID_dict = {key: value for value, key in enumerate(ncID)}

tokenizer = AutoTokenizer.from_pretrained("./annotation/models/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("./annotation/models/DNABERT-2-117M", trust_remote_code=True)
model_cpu = AutoModel.from_pretrained("./annotation/models/DNABERT-2-117M", trust_remote_code=True)

def BERT_embedding(filename, device):
    global model, model_cpu, tokenizer, ncID
    device = torch.device(device)
    model = model.to(device)
    rec = SeqIO.parse(filename, 'fasta')
    IDs, seqs = [], []
    for record in rec:
        IDs.append(record.id)
        seqs.append(str(record.seq))
    
    idx4emb = [idx for idx, id in enumerate(IDs) if id not in ncID]

    if len(idx4emb) != 0:
        res, names = [], []
        for idx in idx4emb:
            id, value = IDs[idx], seqs[idx]
            inputs = tokenizer(value, return_tensors = 'pt')["input_ids"]
            if len(value) > 20000:
                if len(value) > 100000:
                    hidden_states = torch.rand(1, 326, 768) # ensure 768
                else:
                    hidden_states = model_cpu(inputs)[0].detach()
            else:
                inputs = inputs.to(device)
                hidden_states = model(inputs)[0].detach().cpu() # [1, sequence_length, 768]
            
            # embedding with mean pooling
            embedding_mean = torch.mean(hidden_states[0], dim=0)
            res.append(embedding_mean)
            names.append(id)
        
        res_tensor = torch.stack(res)
        lnc_emb_df1 = pd.DataFrame(res_tensor)
        lnc_emb_df1['ncID'] = names
        lnc_emb_df1 = lnc_emb_df1[lnc_emb_df1.columns.tolist()[-1:] + lnc_emb_df1.columns.tolist()[:-1]]
        
        idx_exist = [id for id in IDs if id in ncID]
        if len(idx_exist) != 0:
            lnc_emb_df2 = ncid2fea.loc[ncid2fea['ncID'].isin(idx_exist)]
            lnc_emb_df = pd.concat([lnc_emb_df1, lnc_emb_df2], axis=0)
        else:
            lnc_emb_df = lnc_emb_df1
    
    else:
        lnc_emb_df = ncid2fea.loc[ncid2fea['ncID'].isin([id for id in IDs if id in ncID])]
    
    lnc_emb_df.reset_index(drop=True, inplace=True)
    return lnc_emb_df


