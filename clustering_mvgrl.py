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
from MVGRL_mode_multiscale import GConv, Encoder


def main():
    #datasets = ['PubMed', 'Cora', 'Citeseer']
    datasets = ['Cora']
    device_ids = {'data':0, 'encoder1':1, 'encoder2':2, 'projector':0, 'contrast':0}
    data_eps = {'PubMed':1e-2, 'Cora':1e-5, 'Citeseer':1e-5}
    data_scales = {'PubMed': 4, 'Cora':8, 'Citeseer':8}
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
