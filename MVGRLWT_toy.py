import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, from_predefined_split, LREvaluator
from GCL.models import DualBranchContrast, SingleBranchContrast
from GCL.augmentors.functional import drop_feature
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
from augmentations.diffusionfilters import DiffusionAugmentation
from torch.distributions.dirichlet import Dirichlet
import numpy as np
import yaml
import os
import random
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

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.PReLU(hidden_dim),
        #nn.Dropout(0.5),
        #nn.Linear(hidden_dim, hidden_dim),
        #nn.BatchNorm1d(hidden_dim),
        #nn.PReLU(hidden_dim),
        )

    def forward(self, x):
       return self.mlp(x)

class GConvMultiScale(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nm_scale=8):
        super(GConvMultiScale, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nm_scale = nm_scale
        for i in range(self.nm_scale+1):
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
        features_T = torch.stack(features_T, 1)
        #coeff = self.mixingdistn.sample().to(x.device)
        #coeff = torch.softmax(self.mixing, 0)
        #import pdb
        #pdb.set_trace()
        #features = torch.einsum('t n j, t -> n j', features_T, coeff.squeeze())
        #features = features_T.mean(0)
        return features_T

class Encoder(torch.nn.Module):
    def __init__(self, encoder2, input_dim, hidden_dim, diffusion='wavelet', nm_scale=8, eps=1e-4, device_ids=[0,1,2,3]):
        super(Encoder, self).__init__()
        self.device_ids = device_ids
        self.hidden_dim = hidden_dim
        #self.encoder1 = encoder1.to(device_ids['encoder1'])
        self.encoder2 = encoder2.to(device_ids['encoder2'])
        self.aug = DiffusionAugmentation(nm_scale, filter_type=diffusion, eps=eps).to(device_ids['data'])
        self.nm_scale = nm_scale
        self.project = torch.nn.Linear(hidden_dim, hidden_dim).to(device_ids['projector'])
        #self.coeff = nn.Parameter(torch.ones(self.nm_scale, 1)*(1.0/self.nm_scale), requires_grad=True).to(device_ids['projector'])
        #self.coeff = torch.nn.ModuleList()
        #self.coeff.append(GCNConv(input_dim, hidden_dim))
        #self.coeff.append(GCNConv(hidden_dim, hidden_dim))
        #self.proj1 = MLP(hidden_dim).to(device_ids['projector'])
        #self.coeff_proj = torch.nn.Linear(hidden_dim*(self.nm_scale+1), nm_scale+1).to(device_ids['projector'])
        #self.coeff.to(device_ids['encoder1'])
        #self.activation = nn.PReLU(hidden_dim).to(device_ids['encoder1'])
        uniform(hidden_dim, self.project.weight)
        #self.agg = nn.LSTM(hidden_dim, batch_first=False, hidden_size=int(hidden_dim/2), num_layers=1, bidirectional=True).to(self.device_ids['projector'])

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
    def plot_graphs(self, x, edge_index, edge_weight=None, role_labels=None):
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['data'])
        edge_index2, edge_weight2 = self.aug(edge_index.to(self.device_ids['data']), edge_weight)
        pdb.set_trace()
        from torch_geometric.data import Data
        from torch_geometric.utils import to_networkx
        import networkx as nx
        for i in range(len(edge_index2)):
            graph = Data(x, edge_index2[i], edge_weight2[i])
            graph = to_networkx(graph, to_undirected=True, remove_self_loops=True)
            cmap = plt.get_cmap('hot')
            x_range = np.linspace(0, 0.8, len(np.unique(role_labels)))
            coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_labels))}
            node_color = [coloring[role_labels[i]] for i in range(len(role_labels))]
            plt.figure()
            nx.draw_networkx(graph, pos=nx.layout.spring_layout(graph), 
                                       node_color=node_color, cmap='hot')
            plt.savefig(f"graph_scale_{i+1}.png")
            plt.close()
    def forward(self, x, edge_index, edge_weight=None, test=False):
        #if edge_weight is not None:
        #    edge_weight = edge_weight.to(self.device_ids['encoder1'])
        #z1 = self.encoder1(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        #coeff = self.get_coeff(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight)
        #z1n = self.encoder1(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['data'])
        edge_index2, edge_weight2 = self.aug(edge_index.to(self.device_ids['data']), edge_weight, test=test)
        #pdb.set_trace()
        #if edge_weight2 is not None:
        #    edge_weight2 = edge_weight2.to(self.device_ids['encoder2'])

        z2 = self.encoder2(x, edge_index2, edge_weight2, test=test, device=self.device_ids['encoder2'])
        #h0, c0 = torch.zeros(2, z2.shape[1], int(self.hidden_dim/2)), torch.zeros(2, z2.shape[1], int(self.hidden_dim/2))
        #out, (hn, cn) = self.agg(z2.to(self.device_ids['projector']), (h0.to(self.device_ids['projector']), c0.to(self.device_ids['projector'])))
        #z2 = out
        #z2 = self.proj1(z2.view(-1, self.hidden_dim).to(self.device_ids['projector'])).view(-1, self.nm_scale+1, self.hidden_dim)
        #coeff_pos = torch.softmax(self.coeff_proj(z2.view(x.size(0), -1).to(self.device_ids['projector'])), 1)
        #z2 = self.proj1(z2.view(-1, self.hidden_dim).to(self.device_ids['projector'])).view(-1, self.nm_scale+1, self.hidden_dim)
        #z2 = self.encoder2(x2, edge_index2, edge_weight2)
        #z2 = z2.mean(0).to(self.device_ids['projector'])
        #coeff = torch.softmax(self.coeff, 0).squeeze()
        #z2 = torch.einsum('n k d, n k -> n d', z2.to(self.device_ids['projector']), coeff_pos)
        #g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        #g2 = self.project(torch.sigmoid(z2.mean(1).squeeze().mean(dim=0, keepdim=True)).to(self.device_ids['projector']))
        z2n, g2 = [], []
        for t in range(self.nm_scale+1):
            g2.append(self.project(torch.sigmoid(z2[t].mean(dim=0, keepdim=True)).to(self.device_ids['projector'])))
            # x_t= torch.cat([x.to(self.device_ids['encoder2']), self.encoder2.onehot[t].repeat(x.size(0), 1).to(self.device_ids['encoder2'])], 1)
            #x2_t = self.encoder2.mlp_layers(x_t)
            z2n_t = self.encoder2.layers[t](*self.corruption(x.to(self.device_ids['encoder2']), edge_index2[t].to(self.device_ids['encoder2']), edge_weight2[t].to(self.device_ids['encoder2'])))
            z2n.append(z2n_t)
        z2n = torch.stack(z2n, 1)
        #z2n = self.proj1(z2n.view(-1, self.hidden_dim).to(self.device_ids['projector'])).view(-1, self.nm_scale+1, self.hidden_dim)
        #out, (hn, cn) = self.agg(z2n.to(self.device_ids['projector']), (h0.to(self.device_ids['projector']), c0.to(self.device_ids['projector'])))
        #z2n = out`
        #coeff_neg = torch.softmax(self.coeff_proj(z2n.view(x.size(0), -1).to(self.device_ids['projector'])), 1)
        #z2n = self.proj1(z2n.view(-1, self.hidden_dim).to(self.device_ids['projector'])).view(-1, self.nm_scale+1, self.hidden_dim)
        g2 = torch.stack(g2, 1)
        #coeffn = self.get_coeff(*self.corruption(x.to(self.device_ids['encoder1']), edge_index.to(self.device_ids['encoder1']), edge_weight))
        #z2n = torch.einsum('n k d, n k -> n d', z2n, coeff_neg)
        #coeff = torch.softmax(self.encoder2.mixing, 0)
        #z2n = torch.einsum('t n j, t -> n j', z2n, coeff.squeeze())
        #z2n = z2n.mean(0)
        #z2 = z2.mean(0)
        return z2.to(self.device_ids['contrast']) , g2.to(self.device_ids['contrast']), z2n.to(self.device_ids['contrast'])
        #return z1.to(self.device_ids['contrast']), z2.to(self.device_ids['contrast']), g1.to(self.device_ids['contrast']), g2.to(self.device_ids['contrast']), z1n.to(self.device_ids['contrast']), z2n.to(self.device_ids['contrast'])


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    #pdb.set_trace()
    #z = encoder_model(data.x, data.edge_index)
    #loss = 0.
    #for i in range(z.shape[1]):
    #    for j in range(z.shape[1]):
    #        if i < j:
    #            loss += contrast_model(h1=z[:,i,:], h2=z[:,j,:])
    #loss = contrast_model(h=z, g=g, hn=zn)
    #z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    #loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss = 0.
    coeff = [1.0 for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(z.shape[1]):
            if i<j:
                loss += contrast_model(h1=z[:,i,:], h2=z[:,j,:], g1=g[:,i,:], g2=g[:,j,:], h3=zn[:,i,:], h4=zn[:,j,:])
        #loss += coeff[scale]* contrast_model(h1=z1, h2=z2[scale], g1=g1, g2=g2[scale], h3=z1n, h4=z2n[scale])
    #loss = loss/z2.shape[0]
    loss.backward()
    optimizer.step()
    return loss.item()


from utils import cluster_graph, unsupervised_evaluate, draw_pca
import matplotlib.pyplot as plt

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
    diffusion = 'wavelet'
    results_path = '/disk/scratch2/asif/workspace/graphNCE/modelsDWT/'
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
            data = dataset
            gconv2 = GConvMultiScale(input_dim=dataset.num_features, hidden_dim=6, num_layers=2, nm_scale=data_scales[dataname])
            encoder_model = Encoder(encoder2=gconv2, diffusion=diffusion, input_dim=dataset.num_features, hidden_dim=6, device_ids=device_ids, nm_scale=data_scales[dataname], eps=data_eps[dataname])
            contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])
            #contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])
            optimizer = Adam(encoder_model.parameters(), lr=0.001)
            #encoder_model.plot_graphs(data.x, data.edge_index, edge_weight=None, role_labels=role_id)
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
