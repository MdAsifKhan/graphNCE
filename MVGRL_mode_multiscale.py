import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
from augmentations.diffusionfilters import DiffusionAugmentation
from torch.distributions.dirichlet import Dirichlet
import numpy as np
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


class GConvMultiScale(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nm_scale):
        super(GConvMultiScale, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nm_scale = nm_scale
        self.aug = DiffusionAugmentation(nm_scale)
        self.activation = nn.PReLU(hidden_dim)
        #self.mixingdistn = Dirichlet(torch.tensor([0.5]*self.nm_scale))
        self.mixing = nn.Parameter(torch.ones(self.nm_scale, 1)*(1/self.nm_scale), requires_grad=True)
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
        t = torch.randint(0, self.nm_scale, (1,))
        #import pdb
        #pdb.set_trace()
        features_T = []
        for t in range(self.nm_scale):
            features_t = self.features(x, edge_index_T[t], edge_weight_T[t])
            features_T.append(features_t)
        features_T = torch.stack(features_T)
        #coeff = self.mixingdistn.sample().to(x.device)
        coeff = torch.softmax(self.mixing, 0)
        #import pdb
        #pdb.set_trace()
        features = torch.einsum('t n j, t -> n j', features_T, coeff.squeeze())
        return features, edge_index_T[t], edge_weight_T[t]

class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, hidden_dim, nm_scale=8, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.device_ids = device_ids
        self.encoder1 = encoder1.to(device_ids['encoder1'])
        self.encoder2 = encoder2.to(device_ids['encoder2'])
        self.nm_scale = nm_scale
        self.project = torch.nn.Linear(hidden_dim, hidden_dim).to(device_ids['projector'])
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = edge_weight.to(device_ids['encoder1'])
        z1 = self.encoder1(x.to(device_ids['encoder1']), edge_index.to(device_ids['encoder1']), edge_weight)
        z1n = self.encoder1(*self.corruption(x, edge_index, edge_weight))
        if edge_weight is not None:
            edge_weight = edge_weight.to(device_ids['encoder2'])
        z2, edge_index2, edge_weight2 = self.encoder2(x.to(device_ids['encoder2']), edge_index.to(device_ids['encoder2']), edge_weight)
        #z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)).to(device_ids['projector']))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)).to(device_ids['projector']))
        z2n = self.encoder2.features(*self.corruption(x.to(device_ids['encoder2']), edge_index2, edge_weight2))
        return z1.to(device_ids['contrast']), z2.to(device_ids['contrast']), g1.to(device_ids['contrast']), g2.to(device_ids['contrast']), z1n.to(device_ids['contrast']), z2n.to(device_ids['contrast'])


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data, device_ids):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    datasets = ['Cora', 'Citeseer', 'PubMed']
    device_ids = {'data':0, 'encoder1':0, 'encoder2':1, 'projector':2, 'contrast':3}
    start_seed = 42
    nm_trials = 50
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsDWT/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results = {'Cora': {'F1Ma':None, 'F1Mi':None, 'Acc':None}, 
                    'Citeseer': {'F1Ma':None, 'F1Mi':None, 'Acc':None}, 
                        'PubMed': {'F1Ma':None, 'F1Mi':None, 'Acc':None}}
    for data in datasets:
        print(f'Training for Dataset {data}')
        F1Ma, F1Mi, acc = [], [], []
        for i in range(nm_trials):
            seed = start_seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            device = torch.device('cuda:7')
            path = osp.join(osp.expanduser('~'), 'datasets')
            dataset = Planetoid(path, name=data, transform=T.NormalizeFeatures())
            data = dataset[0]
            gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
            gconv2 = GConvMultiScale(input_dim=dataset.num_features, hidden_dim=512, num_layers=2, nm_scale=8)
            encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, hidden_dim=512, device_ids=device_ids)
            contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

            optimizer = Adam(encoder_model.parameters(), lr=0.001)

            with tqdm(total=200, desc='(T)') as pbar:
                for epoch in range(1, 201):
                    loss = train(encoder_model, contrast_model, data, optimizer)
                    pbar.set_postfix({'loss': loss})
                    pbar.update()

            test_result = test(encoder_model, data)
            F1Ma.append(test_result["macro_f1"])
            F1Mi.append(test_result["micro_f1"])
            acc.append(test_result["acc"])

            print(f'(E): Trial= {i+1} Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, Acc={test_result["accuracy"]}')
            torch.save({'encoder1':gconv1.state_dict(), 'encoder2':gconv2.state_dict(), 
                            'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, f'{results_path}/model200.pt')

        F1Ma = np.array(F1Ma)
        F1Mi = np.array(F1Mi)
        acc = np.array(acc)
        print(f'Data {data} Mean F1Ma= {F1Ma.mean()} Std F1Ma={F1Ma.std()}')
        print(f'Data {data} Mean F1Mi= {F1Mi.mean()} Std F1Mi={F1Mi.std()}')
        print(f'Data {data} Mean Acc= {acc.mean()} Std Acc={acc.std()}')
        results[data]['F1Ma'] = F1Ma
        results[data]['F1Mi'] = F1Mi
        results[data]['Acc'] = acc
    with open(f'{results_path}DWTmetrics.yaml', 'w') as f:
        yaml.dump(results)

if __name__ == '__main__':
    main()
