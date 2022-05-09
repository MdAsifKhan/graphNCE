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
from GCL.augmentors.functional import drop_feature
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
from augmentations.diffusionfilters import DiffusionAugmentation
from torch.distributions.dirichlet import Dirichlet
import numpy as np
import yaml
import os
import random

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
    def __init__(self, input_dim, hidden_dim, num_layers, nm_scale=8):
        super(GConvMultiScale, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nm_scale = nm_scale
        for i in range(self.nm_scale):
            self.layers.append(GConv(input_dim, hidden_dim, num_layers))
        #self.aug = DiffusionAugmentation(nm_scale, eps=eps)
        #self.activation = nn.PReLU(hidden_dim)
        #self.mixingdistn = Dirichlet(torch.tensor([0.5]*self.nm_scale))
        #self.mixing = nn.Parameter(torch.ones(self.nm_scale, 1)*(1.0/nm_scale), requires_grad=True)
        #self.onehot = torch.eye(self.nm_scale)
        #self.mlp_layers = torch.nn.Sequential(nn.Linear(input_dim+nm_scale, hidden_dim),
        #                                    nn.BatchNorm1d(hidden_dim),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hidden_dim, input_dim))
        #for i in range(num_layers):
        #    if i == 0:
        #        self.layers.append(GCNConv(input_dim, hidden_dim))
        #    else:
        #        self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def features(self, x, edge_index, edge_weight=None, test=False):
        z = x
        #if not test:
            #x = x + 0.001* torch.randn(*x.shape).to(x.device)
            #x = self.mlp_layers(x)
            #x = drop_feature(x, 0.25)
            #x = x[:, torch.randperm(x.size(1))]
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

    def forward(self, x, edge_index, edge_weight=None, test=False, device='cuda:0'):
        #edge_index_T, edge_weight_T = self.aug(edge_index.to('cuda:0'), edge_weight.to('cuda:0'))
        #idx = torch.randint(0, self.nm_scale, (1,))
        #import pdb
        #pdb.set_trace()
        nm_scale = len(edge_index)
        features_T = []
        x = x.to(device)
        for t in range(nm_scale):
            #if not test:
            #    x = x + 0.01* torch.randn(*x.shape).to(x.device)
            #    x = drop_feature(x, 0.1)
            #x_t= torch.cat([x, self.onehot[t].repeat(x.size(0), 1).to(x.device)], 1)
            #x = self.mlp_layers(x_t)
            if edge_weight[t] is not None:
                edge_weight[t] = edge_weight[t].to(device)
            #if not test:
            #    x = x[:, torch.randperm(x.size(1))]
            #features_t = self.features(x, edge_index[t].to(device), edge_weight[t], test)
            features_t = self.layers[t](x, edge_index[t].to(device), edge_weight[t].to(device))
            #import pdb
            #pdb.set_trace()
            features_T.append(features_t)
        features_T = torch.stack(features_T)
        #coeff = self.mixingdistn.sample().to(x.device)
        #coeff = torch.softmax(self.mixing, 0)
        #import pdb
        #pdb.set_trace()
        #features = torch.einsum('t n j, t -> n j', features_T, coeff.squeeze())
        #features = features_T.mean(0)
        return features_T

class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, input_dim, hidden_dim, diffusion='wavelet', nm_scale=8, eps=1e-4, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.device_ids = device_ids
        self.hidden_dim = hidden_dim
        self.encoder1 = encoder1.to(device_ids['encoder1'])
        self.encoder2 = encoder2.to(device_ids['encoder2'])
        self.aug = DiffusionAugmentation(nm_scale, filter_type=diffusion, eps=eps).to(device_ids['data'])
        self.nm_scale = nm_scale
        self.project = torch.nn.Linear(hidden_dim, hidden_dim).to(device_ids['projector'])
        #self.coeff = torch.nn.ModuleList()
        #self.coeff.append(GCNConv(input_dim, hidden_dim))
        #self.coeff.append(GCNConv(hidden_dim, hidden_dim))
        #self.coeff_proj = torch.nn.Linear(hidden_dim, nm_scale).to(device_ids['encoder1'])
        #self.coeff.to(device_ids['encoder1'])
        #self.activation = nn.PReLU(hidden_dim).to(device_ids['encoder1'])
        uniform(hidden_dim, self.project.weight)
        self.agg = nn.LSTM(hidden_dim, batch_first=False, hidden_size=int(hidden_dim/2), num_layers=1, bidirectional=True).to(self.device_ids['projector'])

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

    def forward(self, x, edge_index, edge_weight=None, test=False):
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['encoder1'])
        z1 = self.encoder1(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        #coeff = self.get_coeff(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        z1n = self.encoder1(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['data'])
        edge_index2, edge_weight2 = self.aug(edge_index.to(self.device_ids['data']), edge_weight, test=test)
        #if edge_weight2 is not None:
        #    edge_weight2 = edge_weight2.to(self.device_ids['encoder2'])

        z2 = self.encoder2(x, edge_index2, edge_weight2, test=test, device=self.device_ids['encoder2'])
        h0, c0 = torch.zeros(2, z2.shape[1], int(self.hidden_dim/2)), torch.zeros(2, z2.shape[1], int(self.hidden_dim/2))
        out, (hn, cn) = self.agg(z2.to(self.device_ids['projector']), (h0.to(self.device_ids['projector']), c0.to(self.device_ids['projector'])))
        z2 = out
        
        #z2 = self.encoder2(x2, edge_index2, edge_weight2)
        #z2 = z2.mean(0).to(self.device_ids['projector'])
        #z2 = torch.einsum('k n d, n k -> n d', z2, coeff.to(self.device_ids['encoder2']))
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        #g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z2n, g2 = [], []
        for t in range(self.nm_scale):
            g2.append(self.project(torch.sigmoid(z2[t].mean(dim=0, keepdim=True)).to(self.device_ids['projector'])))
            # x_t= torch.cat([x.to(self.device_ids['encoder2']), self.encoder2.onehot[t].repeat(x.size(0), 1).to(self.device_ids['encoder2'])], 1)
            #x2_t = self.encoder2.mlp_layers(x_t)
            z2n_t = self.encoder2.layers[t](*self.corruption(x.to(self.device_ids['encoder2']), edge_index2[t].to(self.device_ids['encoder2']), edge_weight2[t].to(self.device_ids['encoder2'])))
            z2n.append(z2n_t)
        z2n = torch.stack(z2n)
        out, (hn, cn) = self.agg(z2n.to(self.device_ids['projector']), (h0.to(self.device_ids['projector']), c0.to(self.device_ids['projector'])))
        z2n = out  
        g2 = torch.stack(g2)
        #coeffn = self.get_coeff(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        #z2n = torch.einsum('k n d, n k -> n d', z2n, coeffn.to(self.device_ids['encoder2']))
        #coeff = torch.softmax(self.encoder2.mixing, 0)
        #z2n = torch.einsum('t n j, t -> n j', z2n, coeff.squeeze())
        #z2n = z2n.mean(0).to(self.device_ids['projector'])
        return z1.to(self.device_ids['contrast']), z2.to(self.device_ids['contrast']), g1.to(self.device_ids['contrast']), g2.to(self.device_ids['contrast']), z1n.to(self.device_ids['contrast']), z2n.to(self.device_ids['contrast'])


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    #loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss = 0.
    coeff = [1. for i in range(z2.shape[0])]
    for scale in range(z2.shape[0]):
        loss += coeff[scale]* contrast_model(h1=z1, h2=z2[scale], g1=g1, g2=g2[scale], h3=z1n, h4=z2n[scale])
    #loss = loss/z2.shape[0]
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index, test=True)
    z = z1 + z2.mean(0)
    #split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    split = from_predefined_split(data)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    #datasets = ['Cora', 'Citeseer', 'PubMed']
    datasets = ['Cora']
    device_ids = {'data':4, 'encoder1':5, 'encoder2':6, 'projector':7, 'contrast':4}
    data_eps = {'PubMed':1e-4, 'Cora':1e-4, 'Citeseer':1e-6}
    data_scales = {'PubMed': 4, 'Cora':12, 'Citeseer':8}
    nm_trials = 1
    diffusion = 'wavelet'
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsDWT/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results = {'Cora': {'F1Ma':None, 'F1Mi': None}, 'Citeseer':{'F1Ma':None, 'F1Mi': None}, 'PubMed':{'F1Ma':None, 'F1Mi': None}}
    for dataname in datasets:
        print(f'Training on {dataname}')        
        #seed = 42
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        path = osp.join(osp.expanduser('~'), 'datasets')
        dataset = Planetoid(path, name=dataname, transform=T.NormalizeFeatures())
        data = dataset[0]
        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        gconv2 = GConvMultiScale(input_dim=dataset.num_features, hidden_dim=512, num_layers=2, nm_scale=data_scales['Cora'])
        encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, diffusion=diffusion, input_dim=dataset.num_features, hidden_dim=512, device_ids=device_ids, nm_scale=data_scales[dataname], eps=data_eps[dataname])
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        optimizer = Adam(encoder_model.parameters(), lr=0.001)

        with tqdm(total=300, desc='(T)') as pbar:
            for epoch in range(1, 301):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        torch.save({'encoder1':gconv1.state_dict(), 'encoder2':gconv2.state_dict(), 
		'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, f'{results_path}/model_{dataname}_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}.pt'.replace('.00', ''))
        F1Ma, F1Mi = [], []
        for i in range(nm_trials):
            test_result = test(encoder_model, data)
            F1Ma.append(test_result["macro_f1"])
            F1Mi.append(test_result["micro_f1"])
            print(f'(E): Trial= {i+1} Best test Accuracy={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

        #F1Ma = np.array(F1Ma)
        #F1Mi = np.array(F1Mi)
            
        print(f'Data {dataname} Mean F1Ma= {np.mean(F1Ma)} Std F1Ma={np.std(F1Ma)}')
        print(f'Data {dataname} Mean Acc= {np.mean(F1Mi)} Std Acc={np.std(F1Mi)}')
        results[dataname]['F1Ma'] = [f'{el:.4f}' for el in F1Ma]
        results[dataname]['F1Mi'] = [f'{el:.4f}' for el in F1Mi]
    with open(f'{results_path}DWTmetrics.yaml', 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()
