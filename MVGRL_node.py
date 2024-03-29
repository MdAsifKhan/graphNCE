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
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
import numpy as np
import os
import yaml
import pdb

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

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    #import pdb
    #pdb.set_trace()
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2
    #split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    #split = from_predefined_split(data)
    results_all = []
    for j in range(data.train_mask.shape[1]):
        indices = torch.arange(data.num_nodes)
        split = {'train': indices[data.train_mask[:,j]],
                   'valid': indices[data.val_mask[:,j]],
                    'test': indices[data.test_mask[:,j]]
                }
        result = LREvaluator()(z, data.y, split)
        results_all.append(result)
    return results_all


def main():
    datasets = ['chameleon', 'squirrel']
    #datasets = ['Cornell', 'Texas', 'Wisconsin']
    device_ids = {'data':0, 'encoder1':0, 'encoder2':1, 'projector':2, 'contrast':3}
    nm_trials = 1
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsMVGRL/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    #results = {'Cornell': {'F1Ma':None, 'F1Mi':None}, 
    #                'Texas': {'F1Ma':None, 'F1Mi':None}, 
    #                    'Wisconsin': {'F1Ma':None, 'F1Mi':None}}
    results = {'chameleon': {'F1Ma':None, 'F1Mi':None}, 
		    'crocodile':{'F1Ma':None, 'F1Mi':None},
                     'squirrel': {'F1Ma':None, 'F1Mi':None}
                }
    for dataname in datasets:
        print(f'Training for Dataset {dataname}')
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        path = osp.join(osp.expanduser('~'), 'datasets')
        #dataset = Planetoid(path, name=dataname, transform=T.NormalizeFeatures())
        #dataset = WebKB(path, name=dataname, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(path, name=dataname, transform=T.NormalizeFeatures())
        data = dataset[0]
        #pdb.set_trace()
        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2, eps=1e-3)
        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512, device_ids=device_ids)
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        optimizer = Adam(encoder_model.parameters(), lr=0.001)

        with tqdm(total=300, desc='(T)') as pbar:
            for epoch in range(1, 301):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        torch.save({'encoder1':gconv1.state_dict(), 'encoder2':gconv2.state_dict(), 
                        'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, f'{results_path}/model_{dataname}.pt')
        F1Ma, F1Mi = [], []
        for i in range(nm_trials):
            test_result_all = test(encoder_model, data)
            F1Ma = [test_result["macro_f1"] for test_result in test_result_all]
            F1Mi = [test_result["micro_f1"] for test_result in test_result_all]
            #F1Ma.append(test_result["macro_f1"])
            #F1Mi.append(test_result["micro_f1"])
            #print(f'(E): Trial= {i+1} Best test Accuracy={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

        #F1Ma = np.array(F1Ma)
        #F1Mi = np.array(F1Mi)
        print(f'Data {dataname} Mean F1Ma= {np.mean(F1Ma)} Std F1Ma={np.std(F1Ma)}')
        print(f'Data {dataname} Mean Acc= {np.mean(F1Mi)} Std Acc={np.std(F1Mi)}')
        results[dataname]['F1Ma'] = [f'{el:.4f}' for el in F1Ma]
        results[dataname]['F1Mi'] = [f"{el:.4f}" for el in F1Mi]
        with open(f'{results_path}MVGRLmetrics.yaml', 'w') as f:
            yaml.dump(results, f)

if __name__ == '__main__':
    main()
