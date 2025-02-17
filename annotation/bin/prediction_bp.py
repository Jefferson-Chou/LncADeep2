from .gnn_feat_bp import *
import configparser
import subprocess
import shutil 

tarName_revdict = {value:key for key, value in tarName_dict.items()}

config = configparser.ConfigParser()
config.hidden_size = [300, 128, 64]
config.learning_rate = 0.001
config.epochs = 200
config.batch_size = 128
config.dropout = 0.05

# model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size[0], config.hidden_size[1])
        self.fc2 = nn.Linear(config.hidden_size[1], config.hidden_size[2])
        self.fc3 = nn.Linear(config.hidden_size[2], 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# predict function-related proteins
def predict(pred_file, ensembl_id, dev):
    global model
    with open(pred_file, 'rb') as f:
        inputs, edge_label_index = pickle.load(f) 
    prolist = edge_label_index[1].tolist()
    prolist = [tarName_revdict[i] for i in prolist]
    device = torch.device(dev)
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load('./annotation/models/finetune_model_90_0.9817.pt'))
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.cpu().round().detach().numpy()
    pos_idx = [idx for idx, pred in enumerate(pred) if pred == [1.]]
    pos_pro = [prolist[idx] for idx in pos_idx]
    pos_pro_prob = [outputs.cpu().numpy().tolist()[idx] for idx in pos_idx]
    pos_pro_prob = [prob[0] for prob in pos_pro_prob]
    pos_pro_df = pd.DataFrame({'protein': pos_pro, 'probability': pos_pro_prob})
    pos_pro_df.sort_values(by='probability', ascending=False, inplace=True)
    pos_pro_df.reset_index(drop=True, inplace=True)
    
    return pos_pro_df

def pred_go(filename, dev, r_thread = 10, anno_out = './annotation/output/'):
    export_emb(filename, dev)

    for file in os.listdir('./annotation/tmp/'):
        if file.endswith('.pkl'):
            ensembl_id = file.split('_')[2].split('.')[0]
            pos_pro_df = predict(f'./annotation/tmp/{file}', ensembl_id, dev)
            pos_pro_df.to_csv(f'./annotation/tmp/{ensembl_id}_pred.csv', index=False)

    subprocess.call(['Rscript', './annotation/bin/GO_enrich.r', '-n', str(r_thread), '-d', anno_out])
     
