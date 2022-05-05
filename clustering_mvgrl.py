import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, from_predefined_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
import numpy as np
import os
import yaml

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1.to(device_ids['encoder1'])
        self.encoder2 = encoder2.to(device_ids['encoder2'])
        self.augmentor = augmentor
        self.device_ids = device_ids
        self.project = torch.nn.Linear(hidden_dim, hidden_dim).to(device_ids['projector'])
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        if edge_weight1 is not None:
            edge_weight1 = edge_weight1.to(self.device_ids['encoder1'])
        if edge_weight2 is not None:
            edge_weight2 = edge_weight2.to(self.device_ids['encoder2'])

        z1 = self.encoder1(x1.to(self.device_ids['encoder1']), edge_index1.to(self.device_ids['encoder1']), edge_weight1)
        z2 = self.encoder2(x2.to(self.device_ids['encoder2']), edge_index2.to(self.device_ids['encoder2']), edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        z1n = self.encoder1(*self.corruption(x1.to(self.device_ids['encoder1']), edge_index1.to(self.device_ids['encoder1']), edge_weight1))
        z2n = self.encoder2(*self.corruption(x2.to(self.device_ids['encoder2']), edge_index2.to(self.device_ids['encoder2']), edge_weight2))
        return z1.to(self.device_ids['contrast']), z2.to(self.device_ids['contrast']), g1.to(self.device_ids['contrast']), g2.to(self.device_ids['contrast']), z1n.to(self.device_ids['contrast']), z2n.to(self.device_ids['contrast'])


def main():
    #datasets = ['PubMed', 'Cora', 'Citeseer']
    datasets = ['Cora']
    device_ids = {'data':0, 'encoder1':1, 'encoder2':2, 'projector':0, 'contrast':0}
    data_eps = {'PubMed':1e-2, 'Cora':1e-5, 'Citeseer':1e-5}
    data_scales = {'PubMed': 4, 'Cora':8, 'Citeseer':8}
    nm_trials = 50
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsMVGRL/'
    if not os.path.exists(results_path):
        assert 0,f'Path not found {results_path}'

    for dataname in datasets:
        print(f'Testing for Dataset {dataname}')        
        path = osp.join(osp.expanduser('~'), 'datasets')
        dataset = Planetoid(path, name=dataname, transform=T.NormalizeFeatures())
        data = dataset[0]
        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2, eps=1e-3)
        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        state = torch.load(f'{results_path}/model_{dataname}.pt')
        gconv1.load_state_dict(state['encoder1'])
        gconv2.load_state_dict(state['encoder2'])

        encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512, device_ids=device_ids)
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])


        #state = torch.load(f'{results_path}/model_{dataname}.pt')
        #gconv1.load_state_dict(state['encoder1'])
        #gconv2.load_state_dict(state['encoder2'])
        contrast_model.load_state_dict(state['contrast'])

        encoder_model.eval()
        z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
        z = z1 + z2
        split = from_predefined_split(data)

        from sklearn.cluster import KMeans
        import sklearn.metrics as sklm

        y = data.y
        km = KMeans(n_clusters=y.max().item()+1)
        prediction = km.fit(z.detach().cpu().numpy())
        y_pred = prediction.labels_
        print(f'NMI Score {sklm.adjusted_mutual_info_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')
        print(f'ARI Score {sklm.adjusted_rand_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')

if __name__ == '__main__':
    main()
