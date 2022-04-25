from models.model import ContrastReprsn
from torch.optim import Adam, AdamW
import GCL.losses as L
import wandb
from GCL.eval import get_split, LREvaluator

class GraphCRLTrainer:
	def __init__(self, config):
		self.config = config

		self.model = ContrastReprsn(self.config)
		self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L')
		self.optim = Adam(self.model.parameters(), lr=self.config['optim']['lr'])
		wandb.init(project='graphNCE', entity="randomwalker")

	def update(self, dataset):
		self.model.train()
		self.optim.zero_grad()
		if self.config['resume']:
			start_epoch = self.config['resume_epoch']
		for epoch in self.config['epochs']:
			self.train(dataloader)
			if (epoch + 1) % self.config['test_every'] == 0:
				self.test()
			if (epoch + 1) % self.config['save_every'] == 0:
				self.save(epoch)

	def train(self, dataset):
		epoch_loss = 0
		if self.config['batch_wise']:
			data = DataLoader(dataset, self.config['batch_size'])
		else:
			data = DataLoader(dataset, 1)
			for i, data_batch in enumerate(data):
				data_batch = data_batch.to(self.config['device'])
				if data_batch.x is None:
					data_batch.x = torch.ones((data_batch.batch.size(0), 1), dtpe=torch.float32, device=data_batch.batch.device)

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
			z, _, _ = self.loader(data.x, data.edge_index, data.edge_attr)
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