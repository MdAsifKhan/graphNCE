from torch.utils.data import Dataset
from torch.geometric.datasets import Planetoid




class GraphData(Dataset):
	def __init__(name):
		self.name = dataname
		self.data = []

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		raise NotImplementedError