import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, from_predefined_split, SVMEvaluator
from GCL.models import DualBranchContrast, SingleBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from augmentations.diffusionfilters import DiffusionAugmentation
import pdb
import os
import numpy as np
import yaml
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, batch, x, edge_index, edge_weight=None):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g

class GConvMultiScale(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nm_scale=8):
        super(GConvMultiScale, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nm_scale = nm_scale
        for i in range(self.nm_scale+1):
            self.layers.append(GConv(input_dim, hidden_dim, num_layers))

    def forward(self, batch, x, edge_index, edge_weight=None, test=False):
        nm_scale = len(edge_index)
        z_T, g_T = [],  []
        for t in range(nm_scale):
            #if not test:
            #    x = x + 0.01* torch.randn(*x.shape).to(x.device)
            #    x = drop_feature(x, 0.1)
            #if edge_weight[t] is not None:
            #    edge_weight[t] = edge_weight[t].to(device)
            z_t, g_t = self.layers[t](batch.to(x.device), x, edge_index[t].to(x.device), edge_weight[t].to(x.device))
            z_T.append(z_t)
            g_T.append(g_t)
        g_T = torch.stack(g_T)
        z_T = torch.stack(z_T)
        return z_T, g_T


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, mlp1, mlp2, nm_scales=8, diffusion='wavelet', eps=1e-4, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.device_ids = device_ids
        self.nm_scales = nm_scales
        self.encoder = encoder.to(device_ids['encoder1'])
        self.mlp1 = mlp1.to(device_ids['projector'])
        self.mlp2 = mlp2.to(device_ids['projector'])
        self.aug2 = DiffusionAugmentation(nm_scales, filter_type=diffusion, eps=eps).to(device_ids['data'])

    def forward(self, x, edge_index, batch, edge_weight=None, test=False):
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['data'])

        edge_index_T, edge_weight_T = self.aug2(edge_index.to(self.device_ids['data']), edge_weight, test=test)
        z_T, g_T = self.encoder(batch, x.to(self.device_ids['encoder1']), edge_index_T, edge_weight_T)
        h1_s = [self.mlp1(h.to(self.device_ids['projector'])).to(self.device_ids['contrast']) for h in z_T]
        g1_s = [self.mlp2(g.to(self.device_ids['projector'])).to(self.device_ids['contrast']) for g in g_T]
        return h1_s, g1_s


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        h_T, g_T = encoder_model(data.x, data.edge_index, data.batch)
        loss = 0.
        batch = data.batch.to(h_T[0].device)
        for scale in range(len(h_T)):
            loss += contrast_model(h=h_T[scale], g=g_T[scale], batch=batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _,  g1 = encoder_model(data.x, data.edge_index, data.batch)
        g1 = torch.stack(g1).mean(0)
        x.append(g1)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.6, test_ratio=0.2)
    #split = from_predefined_split(data)
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    clf = svm.SVC(kernel='linear', random_state=42)
    clf_search = GridSearchCV(clf, params, cv=10, scoring='accuracy')
    clf_search.fit(x.detach().cpu().numpy(), y.numpy())
    #result = SVMEvaluator(linear=True)(x, y, split)
    return clf_search.cv_results_['mean_test_score'], clf_search.cv_results_['std_test_score']


def main():
    datasets = ['PTC_MR']
    device_ids = {'data':4, 'encoder1':5, 'projector':7, 'contrast':4}
    data_eps = {'PTC_MR':1e-4}
    data_scales = {'PTC_MR': 4}
    nm_trials = 1
    diffusion = 'wavelet'
    path = osp.join(osp.expanduser('~'), 'datasets')
    results = {'PTC_MR': {'F1Ma':None, 'F1Mi': None}}
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsGraphMVGRL'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for dataname in datasets:
        dataset = TUDataset(path, name=dataname)
        dataloader = DataLoader(dataset, batch_size=128)
        input_dim = max(dataset.num_features, 1)
        

        gcn = GConvMultiScale(input_dim=input_dim, hidden_dim=512, num_layers=2, nm_scale=data_scales[dataname])
        mlp1 = FC(input_dim=512, output_dim=512)
        mlp2 = FC(input_dim=512 * 2, output_dim=512)
        encoder_model = Encoder(encoder=gcn, mlp1=mlp1, mlp2=mlp2, nm_scales=data_scales[dataname], 
                                        diffusion=diffusion, eps=data_eps[dataname], device_ids=device_ids)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        optimizer = Adam(encoder_model.parameters(), lr=0.01)
        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = train(encoder_model, contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        torch.save({'encoder':gcn.state_dict(), 'projection1':mlp1.state_dict(), 'projection2':mlp2.state_dict(), 
                    'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, 
            f'{results_path}/model_{dataname}_graph_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}.pt'.replace('.00', ''))
        
        #F1Ma, F1Mi = [], []
        #for i in range(nm_trials):
        test_result = test(encoder_model, dataloader)
        #F1Ma.append(test_result["macro_f1"])
        #F1Mi.append(test_result["micro_f1"])
        #print(f'(E): Trial= {i+1} Best test Accuracy={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

            
        #print(f'Data {dataname} Mean F1Ma= {np.mean(F1Ma)} Std F1Ma={np.std(F1Ma)}')
        print(f'Data {dataname} Mean Acc= {test_result[0]} Std Acc={test_result[1]}')
        #results[dataname]['F1Ma'] = [f'{el:.4f}' for el in F1Ma]
        #results[dataname]['F1Mi'] = [f'{el:.4f}' for el in F1Mi]
        with open(f'{results_path}/{dataname}DWTmetricsGraph.yaml', 'w') as f:
            yaml.dump(results, f)


if __name__ == '__main__':
    main()
