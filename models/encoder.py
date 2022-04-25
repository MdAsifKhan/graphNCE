import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
	def __init__(self, config):

		super(Encoder, self).__init__()
		'''
		config: dictionary of class variables 
		in_dim : input dimension
		z_dim: dimensionality of latent space
		nm_layers: number of graph convolution layers
		'''

		self.config = config
		self.activation = nn.PReLU(self.config['z_dim'])
		self.network = []
		for i in range(self.config['nm_layers']):
			if i == 0:
				self.network.append(GCNConv(self.config['in_dim'], self.config['z_dim']))
			else:
				self.network.append(GCNConv(self.config['in_dim'], self.config['z_dim']))

	def forward(self, x, edge_index, edge_weight=None):
		for layer in self.layers:
			x = layer(x, edge_index, edge_weight)
			x = self.activation(x)
		return x