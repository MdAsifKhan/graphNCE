from trainer import GraphCRLTrainer
from torch_geometric.datasets import TUDataset, Planetoid
import argparse
from torch_geometric.data import DataLoader
import yaml
import torch
import numpy as np
import torch_geometric.transforms as T


def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--cuda', type=bool, default=True)

opt = parser.parse_args()

config = get_config(opt.config)


torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config['device'] = torch.device(f"cuda:{opt.gpu_id}" if opt.cuda else 'cpu')

#dataset = TUDataset(config['data_path'], name='PTC_MR')
dataset = Planetoid(config['data_path'], name='Cora', transform=T.NormalizeFeatures())

in_dim = max(dataset.num_features, 1)
config['model']['encoder']['in_dim'] = in_dim

run = GraphCRLTrainer(config)

run.update(dataset)


