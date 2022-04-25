import torch
import torch.nn as nn
import GCL.augmentors as A
from models.encoder import Encoder
 
class ContrastReprsn(nn.Module):
	def __init__(self, config):

		super(ContrastReprsn, self).__init__()
		'''
		config: dictionary of class variables 
		in_dim : input dimension
		z_dim: dimensionality of latent space
		nm_layers: number of graph convolution layers
		'''

		self.config = config
		self.encoder = Encoder(self.config['encoder'])
		self.projector = nn.Linear(self.config['z_dim'], self.config['z_dim'])


	def augmentor(self, x, edge_index, edge_weight=None):
		if self.config['augtype'] == 'GraphCL':		
			aug1 = A.Identity()
			aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
									A.NodeDropping(pn=0.1),
									A.FeatureMasking(pf=0.1),
									A.EdgeRemoving(pe=0.1)], 1)
		elif self.config['augtype'] == 'GRACE':
			aug1 = A.Compose([A.EdgeRemoving(pe=0.3). A.FeatureMasking(pf=0.3)])
			aug2 = A.Compose([A.EdgeRemoving(pe=0.3). A.FeatureMasking(pf=0.3)])
		elif self.config['augtype'] == 'transformer':
			raise NotImplementedError
		else:
			pass

		x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
		x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

		return (x1, edge_index1, edge_weight1), (x2, edge_index2, edge_weight2) 

	def forward(self, x, edge_index, edge_weight=None):
		aug1, aug2 = self.augmentor(x, edge_index, edge_weight)
		x1, edge_index1, edge_weight1 = aug1
		x2, edge_index2, edge_weight2 = aug2

		z = self.encoder(x, edge_index, edge_weight)
		z1 = self.encoder(x1, edge_index1, edge_weight1)
		z2 = self.encoder(x2, edge_index2, edge_weight2)

		return z, z1, z2

	def project(self, z):
		return self.project(z)
	