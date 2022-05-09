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


from MVGRL_mode_multiscale import GConv, GConvMultiScale, Encoder
def main():
    #datasets = ['PubMed', 'Cora', 'Citeseer']
    datasets = ['Cora']
    device_ids = {'data':0, 'encoder1':1, 'encoder2':2, 'projector':3, 'contrast':3}
    data_eps = {'PubMed':1e-2, 'Cora':1e-4, 'Citeseer':1e-5}
    data_scales = {'PubMed': 4, 'Cora':12, 'Citeseer':8}
    diffusion = 'wavelet'
    results_path = '/disk/scratch1/asif/workspace/graphNCE/modelsDWT/'

    for dataname in datasets:
        print(f'Testing for Dataset {dataname}')        
        path = osp.join(osp.expanduser('~'), 'datasets')
        dataset = Planetoid(path, name=dataname, transform=T.NormalizeFeatures())
        data = dataset[0]

        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2)
        gconv2 = GConvMultiScale(input_dim=dataset.num_features, hidden_dim=512, num_layers=2, nm_scale=data_scales['Cora'])
        state = torch.load(f'{results_path}/model_{dataname}_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}.pt'.replace('.00', ''))
        gconv1.load_state_dict(state['encoder1']) 
        gconv2.load_state_dict(state['encoder2']) 
        
        encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, diffusion=diffusion, input_dim=dataset.num_features, hidden_dim=512, device_ids=device_ids, nm_scale=data_scales[dataname], eps=data_eps[dataname])
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        #state = torch.load(f'{results_path}/model_{dataname}.pt')
        #encoder_model.encoder1.load_state_dict(state['encoder1'])
        #encoder_model.encoder2.load_state_dict(state['encoder2'])
        contrast_model.load_state_dict(state['contrast'])
        encoder_model.eval()
        z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index, test=True)
        z = z1 + z2.mean(0)
        split = from_predefined_split(data)

        from sklearn.cluster import KMeans, SpectralClustering
        import sklearn.metrics as sklm

        y = data.y
        km = KMeans(n_clusters=y.max().item()+1)
        z = z.detach().cpu().numpy()
        #z = (z - z.mean(0))/(z.std(0)+1e-6)
        #l = 5e1
        #mu = 1.
        #u,s,v = np.linalg.svd(mu*np.eye(z.shape[1]) + l*np.matmul(z.T, z), full_matrices=False)
        #H_inv = np.matmul(v.T, (u/np.expand_dims(s, 0)).T)
        #A_latent = l*np.matmul(np.matmul(z, H_inv), z.T)
        #A_latent = (A_latent - A_latent.min())/(A_latent.min()+A_latent.max())
        #km = SpectralClustering(n_clusters=y.max().item()+1, affinity='precomputed')
        prediction = km.fit(z)
        #prediction = km.fit(A_latent)
        y_pred = prediction.labels_
        print(f'NMI Score {sklm.adjusted_mutual_info_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')
        print(f'ARI Score {sklm.adjusted_rand_score(y.cpu().numpy().reshape(-1), y_pred.reshape(-1))}')
        
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        z_embed = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(z)
        plt.scatter(z_embed[:,0], z_embed[:,1],  c=y.reshape(-1))
        plt.savefig(f'{results_path}/model_{dataname}_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}.png'.replace('.00',''))

if __name__ == '__main__':
    main()
