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


def test(encoder_model, data, role_id, trial=0, out_path='results/'):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, test=True)
    
    labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, z.mean(1))
    hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
    print(f"Trial {trial} \t Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n")
    print(hom, comp, ami, nb_clust, ch, sil)
    x_range = np.linspace(0, 0.9, len(np.unique(role_id)))
    cmap = plt.get_cmap('hot')
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    draw_pca(role_id, z.mean(1), coloring, out_path)
    return hom, comp, ami, nb_clust, ch, sil
    
from data import build_graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

def main():
    datasets = ['house']
    device_ids = {'data':0, 'encoder1':0, 'encoder2':0, 'projector':0, 'contrast':0}
    #coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))} 
    data_eps = {'house': 1e-1, 'varied':1e-5}
    data_scales = {'house':3, 'varied':4}
    data_shapes = {'house': [["house"]]*5, 'varied':[["fan",8]]+[["star",8]]+[["house", 8]]}
    data_wb = {'house':15, 'varied':25}
    nm_trials = 1
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsMVGRL/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    homs, comps, amis, chs, sils = [], [], [], [], []
    for dataname in datasets:
        print(f'Training on {dataname}')        
        #seed = 42
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        for t in range(nm_trials):        
            G, _, _, role_id = build_graph.build_structure(width_basis=data_wb[dataname], basis_type='cycle', list_shapes=data_shapes[dataname], start=0, rdm_basis_plugins=False, add_random_edges=0, plot=True, savefig=False)
            print('nb of nodes in the graph: ', G.number_of_nodes())
            print('nb of edges in the graph: ', G.number_of_edges())
            graph = from_networkx(G)
            x = torch.tensor([[deg] for _, deg in G.degree()], dtype=torch.float)
            dataset = Data(x=x, edge_index=graph.edge_index)

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

            torch.save({'encoder2':gconv2.state_dict(), 
            'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, f'{results_path}/model_{dataname}_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}_trial_{t}.pt'.replace('.00', ''))
        
            hom, comp, ami, nb_clust, ch, sil = test(encoder_model, data, role_id, t, results_path)
            homs.append(hom)
            comps.append(comp)
            amis.append(ami)
            chs.append(ch)
            sils.append(sil)
        print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
        print(str(sum(homs)/nm_trials), str(sum(comps)/nm_trials), str(sum(amis)/nm_trials), str(nb_clust), 
                 str(sum(chs)/nm_trials), str(sum(sils)/nm_trials))

if __name__ == '__main__':
    main()
