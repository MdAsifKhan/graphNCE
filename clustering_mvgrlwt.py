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
from augmentations.diffusionfilters import DiffusionAugmentation
from torch.distributions.dirichlet import Dirichlet
import numpy as np
import yaml
import os
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


class GConvMultiScale(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nm_scale, eps=1e-3):
        super(GConvMultiScale, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nm_scale = nm_scale
        self.aug = DiffusionAugmentation(nm_scale, eps=eps)
        self.activation = nn.PReLU(hidden_dim)
        #self.mixingdistn = Dirichlet(torch.tensor([0.5]*self.nm_scale))
        #self.mixing = nn.Parameter(torch.ones(self.nm_scale, 1)*(1.0/nm_scale), requires_grad=True)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def features(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

    def forward(self, x, edge_index, edge_weight=None):
        edge_index_T, edge_weight_T = self.aug(edge_index, edge_weight)
        #idx = torch.randint(0, self.nm_scale, (1,))
        #import pdb
        #pdb.set_trace()
        features_T = []
        for t in range(self.nm_scale):
            features_t = self.features(x, edge_index_T[t], edge_weight_T[t])
            features_T.append(features_t)
        features_T = torch.stack(features_T)
        #coeff = self.mixingdistn.sample().to(x.device)
        #coeff = torch.softmax(self.mixing, 0)
        #import pdb
        #pdb.set_trace()
        #features = torch.einsum('t n j, t -> n j', features_T, coeff.squeeze())
        #features = features_T.mean(0)
        return features_T, edge_index_T, edge_weight_T

class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, input_dim, hidden_dim, nm_scale=8, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.device_ids = device_ids
        self.encoder1 = encoder1.to(device_ids['encoder1'])
        self.encoder2 = encoder2.to(device_ids['encoder2'])
        self.nm_scale = nm_scale
        self.project = torch.nn.Linear(hidden_dim, hidden_dim).to(device_ids['projector'])
        #self.coeff = torch.nn.ModuleList()
        #self.coeff.append(GCNConv(input_dim, hidden_dim))
        #self.coeff.append(GCNConv(hidden_dim, hidden_dim))
        #self.coeff_proj = torch.nn.Linear(hidden_dim, nm_scale).to(device_ids['encoder1'])
        #self.coeff.to(device_ids['encoder1'])
        #self.activation = nn.PReLU(hidden_dim).to(device_ids['encoder1'])
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    #def get_coeff(self, x, edge_index, edge_weight):
    #    z = x
    #    for conv in self.coeff:
    #        z = conv(z, edge_index, edge_weight)
    #        z = self.activation(z)
    #    z = torch.softmax(self.coeff_proj(z), 0)
    #    return z

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['encoder1'])
        z1 = self.encoder1(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        #coeff = self.get_coeff(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        z1n = self.encoder1(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['encoder2'])
        z2, edge_index2, edge_weight2 = self.encoder2(x.to(self.device_ids['encoder2']), edge_index.to(self.device_ids['encoder2']), edge_weight)
        #z2 = self.encoder2(x2, edge_index2, edge_weight2)
        #z2 = z2.mean(0)
        #z2 = torch.einsum('k n d, n k -> n d', z2, coeff.to(self.device_ids['encoder2']))
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        #g2 = self.project(torch.sigmoid(z2[t].mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        g2, z2n = [], []
        for t in range(self.nm_scale):
            g2.append(self.project(torch.sigmoid(z2[t].mean(dim=0, keepdim=True)).to(self.device_ids['projector'])))
            z2n_t = self.encoder2.features(*self.corruption(x.to(self.device_ids['encoder2']), edge_index2[t], edge_weight2[t]))
            z2n.append(z2n_t)
        z2n = torch.stack(z2n)
        g2 = torch.stack(g2)
        #coeffn = self.get_coeff(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        #z2n = torch.einsum('k n d, n k -> n d', z2n, coeffn.to(self.device_ids['encoder2']))
        #coeff = torch.softmax(self.encoder2.mixing, 0)
        #z2n = torch.einsum('t n j, t -> n j', z2n, coeff.squeeze())
        #z2n = z2n.mean(0)
        return z1.to(self.device_ids['contrast']), z2.to(self.device_ids['contrast']), g1.to(self.device_ids['contrast']), g2.to(self.device_ids['contrast']), z1n.to(self.device_ids['contrast']), z2n.to(self.device_ids['contrast'])


def main():
    datasets = ['PubMed', 'Cora', 'Citeseer']
    #datasets = ['PubMed']
    device_ids = {'data':4, 'encoder1':5, 'encoder2':6, 'projector':7, 'contrast':4}
    data_eps = {'PubMed':1e-2, 'Cora':1e-4, 'Citeseer':1e-5}
    data_scales = {'PubMed': 4, 'Cora':8, 'Citeseer':8}
    nm_trials = 50
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsDWT/'

    for dataname in datasets:
        print(f'Testing for Dataset {dataname}')        
        path = osp.join(osp.expanduser('~'), 'datasets')
        dataset = Planetoid(path, name=dataname, transform=T.NormalizeFeatures())
        data = dataset[0]
        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        gconv2 = GConvMultiScale(input_dim=dataset.num_features, hidden_dim=512, num_layers=2, nm_scale=data_scales[dataname], eps=data_eps[dataname])
        encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, hidden_dim=512, device_ids=device_ids, nm_scale=data_scales[dataname])
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        checkpoint = torch.load(f'{results_path}/model_{dataname}.pt', map_location=device_ids['data'])
        gconv1.load_state_dict(state['encoder1'])
        gconv2.load_state_dict(state['encoder2'])
        contrast_model.load_state_dict(state['contrast'])

        encoder_model.eval()
        z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
        z = z1 + z2.mean()
        split = from_predefined_split(data)

        from sklearn.cluster import KMeans
        import sklearn.metrics as sklm

        y = data.y
        km = KMeans(n_clusters=y.max().item()+1)
        prediction = km.fit(z)
        y_pred = prediction.labels_
        print(f'NMI Score {sklm.adjusted_mutual_info_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')
        print(f'ARI Score {sklm.adjusted_mutual_info_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')

if __name__ == '__main__':
    main()
