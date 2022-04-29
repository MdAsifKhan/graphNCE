from models.model import ContrastReprsn
from torch.optim import Adam, AdamW
import torch
import GCL.losses as L
import wandb
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from tqdm import tqdm
from torch_geometric.data import DataLoader
import GCL.augmentors as A
import os

class GraphCRLTrainer:
	def __init__(self, config):
		self.config = config
		
		#aug1 = A.Identity()
		#aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
		#			A.NodeDropping(pn=0.1),
		#			A.FeatureMasking(pf=0.1),
		#			A.EdgeRemoving(pe=0.1)], 1)
		#self.augmentor = aug1, aug2

		self.model = ContrastReprsn(self.config['model']).to(self.config['device'])
		self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(self.config['device'])
		self.optim = Adam(self.model.parameters(), lr=self.config['optim']['lr'])
		wandb.init(project='graphNCE', entity="randomwalker")

	def update(self, dataset):
		if self.config['resume']:
			start_epoch = self.config['resume_epoch']
		else:
			start_epoch = 0
		with tqdm(total=self.config['epochs'], desc='(T)') as pbar:
			for epoch in range(start_epoch, self.config['epochs']):
				self.train(dataset)
				if (epoch + 1) % self.config['test_every'] == 0:
					self.test(dataset)
				if (epoch + 1) % self.config['save_every'] == 0:
					self.save(epoch)
			pbar.update()
	def train(self, dataset):
		self.model.train()
		epoch_loss = 0
		if self.config['batch_wise']:
			data = DataLoader(dataset, self.config['batch_size'])
		else:
			data = DataLoader(dataset, 1)
		for i, data_batch in enumerate(data):
			data_batch = data_batch[0].to(self.config['device'])
			self.optim.zero_grad()
			if data_batch.x is None:
				data_batch.x = torch.ones((data_batch.batch.size(0), 1), dtype=torch.float32, device=data_batch.batch.device)

			z, z1, z2 = self.model(data_batch.x, data_batch.edge_index, data_batch.edge_attr)
			h1, h2 = [self.model.project(x) for x in [z1, z2]]
			loss = self.contrast_model(h1, h2)
			loss.backward()
			self.optim.step()
			epoch_loss += loss.item()
			wandb.log({"Iteration Loss": loss.item()})
		epoch_loss /= (i+1)
		wandb.log({"Epoch Loss": epoch_loss})

	def test(self, dataset):
		self.model.eval()
		if self.config['batch_wise']:
			data = DataLoader(dataset, self.config['batch_size'])
		else:
			data = DataLoader(dataset, 1)
		z_all, y_all = [], []
		for i, data in enumerate(data):
			data = data[0].to(self.config['device'])
			z, _, _ = self.model(data.x, data.edge_index, data.edge_attr)
			z_all.append(z)
			y_all.append(data.y)
		z_all = torch.cat(z_all, dim=0)
		y_all = torch.cat(y_all, dim=0)
		
		split = get_split(num_samples=z_all.size()[0], train_ratio=0.1, test_ratio=0.8)
		evaluation = LREvaluator()(z_all, y_all, split)
		print(f" Best test F1Micro {evaluation['micro_f1']}, F1Macro {evaluation['macro_f1']}")
		wandb.log({"Test F1 Macro": evaluation['macro_f1'],
					"Test F1 Micro": evaluation['micro_f1']})

	def save(self, epoch):
		model = os.path.join(self.config['results_path'], f"models/model_{epoch+1}.pt")
		torch.save({'model':self.model.state_dict(), 'optim':self.optim.state_dict()}, model)

	def resume(self, checkpoint, device=torch.device('cpu')):
		state = torch.load(checkpoint, map_location=device)
		self.model.load_state_dict(state['model'])
		self.optim.load_state_dict(state['optim'])
