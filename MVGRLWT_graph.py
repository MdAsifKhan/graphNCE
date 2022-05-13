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
        for i in range(self.nm_scale):
            self.layers.append(GConv(input_dim, hidden_dim, num_layers))

    def forward(self, batch, x, edge_index, edge_weight=None, test=False, device='cuda:0'):
        nm_scale = len(edge_index)
        z_T, g_T = [],  []
        x = x.to(device)
        for t in range(nm_scale):
            #if not test:
            #    x = x + 0.01* torch.randn(*x.shape).to(x.device)
            #    x = drop_feature(x, 0.1)
            if edge_weight[t] is not None:
                edge_weight[t] = edge_weight[t].to(device)
            z_t, g_t = self.layers[t](batch, x, edge_index[t].to(device), edge_weight[t])
            z_T.append(features_t)
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
        self.encoder = encoder.to(device_ids['encoder2'])
        self.mlp1 = mlp1.to(device_ids['projection1'])
        self.mlp2 = mlp2.to(device_ids['projection2'])
        self.aug2 = DiffusionAugmentation(nm_scale, filter_type=diffusion, eps=eps).to(device_ids['data'])

    def forward(self, x, edge_index, batch, edge_weight=None, test=False):
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device_ids['encoder1'])

        edge_index_T, edge_weight_T = self.aug2(edge_index.to(self.device_ids['data']), edge_weight, test=test)

        z_T, g_T = self.encoder(batch, x1, edge_index1, edge_weight1)
        h1_s = [self.mlp1(h.to(device_ids['projection1'])) for h in z_T[scale]]
        g1_s = [self.mlp2(g.to(device_ids['projection2'])) for g in g_T[scale]]
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
        for scale in range(len(h1)):
            loss += contrast_model(h1=h_T[scale], g1=g_T[scale], batch=data.batch)
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
        _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1 = g1.mean(0)
        g2 = g2.mean(0)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    #split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    split = from_predefined_split(data)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    datasets = ['PTC_MR']
    device_ids = {'data':4, 'encoder1':5, 'encoder2':6, 'projector':7, 'contrast':4}
    data_eps = {'PTC_MR':1e-4}
    data_scales = {'PTC_MR': 4}

    diffusion = 'wavelet'
    path = osp.join(osp.expanduser('~'), 'datasets')
    results = {'PTC_MR': {'F1Ma':None, 'F1Mi': None}}
    for dataname in datasets:
        dataset = TUDataset(path, name=dataname)
        dataloader = DataLoader(dataset, batch_size=128)
        input_dim = max(dataset.num_features, 1)
        

        gcn2 = GConvMultiScale(input_dim=input_dim, hidden_dim=512, num_layers=2, data_scales[dataname])
        mlp1 = FC(input_dim=512, output_dim=512)
        mlp2 = FC(input_dim=512 * 2, output_dim=512)
        encoder_model = Encoder(encoder=gcn, mlp1=mlp1, mlp2=mlp2, nm_scales=data_scales[dataname], 
                                        diffusion=diffusion, eps=data_eps[dataname], device_ids=device_ids)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device_ids['contrast'])

        optimizer = Adam(encoder_model.parameters(), lr=0.01)

        torch.save({'encoder1':gconv1.state_dict(), 'encoder2':gconv2.state_dict(), 
            'contrast':contrast_model.state_dict(),'optim':optimizer.state_dict()}, 
            f'{results_path}/model_{dataname}_graph_diffusion_{diffusion}_scales_{data_scales[dataname]}_eps_{data_eps[dataname]:.2e}.pt'.replace('.00', ''))
        
        F1Ma, F1Mi = [], []
        for i in range(nm_trials):
            test_result = test(encoder_model, data)
            F1Ma.append(test_result["macro_f1"])
            F1Mi.append(test_result["micro_f1"])
            print(f'(E): Trial= {i+1} Best test Accuracy={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

            
        print(f'Data {dataname} Mean F1Ma= {np.mean(F1Ma)} Std F1Ma={np.std(F1Ma)}')
        print(f'Data {dataname} Mean Acc= {np.mean(F1Mi)} Std Acc={np.std(F1Mi)}')
        results[dataname]['F1Ma'] = [f'{el:.4f}' for el in F1Ma]
        results[dataname]['F1Mi'] = [f'{el:.4f}' for el in F1Mi]
    with open(f'{results_path}DWTmetricsGraph.yaml', 'w') as f:
        yaml.dump(results, f)
        test_result = test(encoder_model, dataloader)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
