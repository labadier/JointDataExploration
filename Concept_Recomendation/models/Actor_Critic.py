import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

from sklearn.metrics.pairwise import cosine_similarity

from glob import glob

#actor critic model
class Actor(torch.nn.Module):
	
	def __init__(self, state_dim: int, action_dim: int,
			  lr_optimizer: float = 0.01):
		super(Actor, self).__init__()

		self.condense_text = torch.nn.Sequential(
			torch.nn.Linear(state_dim >> 1, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 256),
			torch.nn.BatchNorm1d(256), 
			torch.nn.ReLU())

		self.condense_image = torch.nn.Sequential(
			torch.nn.Linear(state_dim >> 1, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 256),
			torch.nn.BatchNorm1d(256), 
			torch.nn.ReLU())

		self.head = torch.nn.Sequential(
			torch.nn.Linear(256, 128),
			torch.nn.BatchNorm1d(128), 
			torch.nn.ReLU(),
			torch.nn.Linear(128, action_dim),
		)

		self.criterion = torch.nn.MSELoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.to(self.device)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_optimizer)
		
	def forward(self, state):

		state = state.to(self.device)
		i = self.condense_image(state[..., :state.shape[-1] >> 1])
		t = self.condense_text(state[..., state.shape[-1] >> 1:])
		return self.head(i + t)
		# return self.model(state.to(self.device))
		# return self.model(torch.cat([state, action], dim = -1).to(self.device)).squeeze(-1)
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
    

class Critic(torch.nn.Module):

	def __init__(self, state_dim: int,
			  lr_optimizer: float = 0.01):
		super(Critic, self).__init__()
		
		self.condense_text = torch.nn.Sequential(
			torch.nn.Linear(state_dim >> 1, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 256),
			torch.nn.BatchNorm1d(256), 
			torch.nn.ReLU())

		self.condense_image = torch.nn.Sequential(
			torch.nn.Linear(state_dim >> 1, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 256),
			torch.nn.BatchNorm1d(256), 
			torch.nn.ReLU())

		self.head = torch.nn.Sequential(
			torch.nn.Linear(256, 128),
			torch.nn.BatchNorm1d(128), 
			torch.nn.ReLU(),
			torch.nn.Linear(128, 1),
		)

		self.criterion = torch.nn.MSELoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.to(self.device)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_optimizer)
		

	def forward(self, state):
		state = state.to(self.device)
		i = self.condense_image(state[..., :state.shape[-1] >> 1])
		t = self.condense_text(state[..., state.shape[-1] >> 1:])
		return self.head(i + t)
	
	def save(self, path):
		torch.save(self.state_dict(), path)


class AdaptationEngine(torch.nn.Module):

	def __init__(self, state_dim: int, 
					action_dim: int,
					action_space_len: int,
					lr_actor: float = 0.001,
					lr_critic: float = 0.0001,
					gamma: float = 0.99,
					buffer_size: int = 512,
					sample_temperature: float = 5.0,
					final_temperature: float = 0.1,
					temperature_decay: float = 0.99,
					**kwargs
					):
		super(AdaptationEngine, self).__init__()

		self.Actor = Actor(state_dim = state_dim, action_dim = action_dim,
							lr_optimizer = lr_actor)
		self.Critic = Critic(state_dim = state_dim, lr_optimizer=lr_critic)
		self.gamma = gamma

		self.buffer_size = buffer_size
		self.buffer = []
		self.temperature = sample_temperature
		self.final_temperature = final_temperature
		self.temperature_decay = temperature_decay
		self.action_space_len = action_space_len

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.max_reward = None


	def push_buffer(self, state, action, reward, next_state, done, masked_indices = None):

		mask = torch.zeros(self.action_space_len)
		mask[masked_indices] = 1

		self.buffer.append({'state': state,
							'action': action, 
							'reward': reward, 
							'next_state': next_state, 
							'end_state': done,
							'masked_index': mask})
		
	def policy(self, state, actions_encode, masking_actions = None):
		
		action_latent = self.Actor.forward(state) #b x action_dim
		preferences = action_latent @ actions_encode.to(action_latent.device).T
		
		
		if masking_actions is not None:
			preferences = preferences +  -1e8*masking_actions.to(preferences.device)
		# preferences = preferences +  -1e8*mask.unsqueeze(0).to(preferences.device)
		preferences -= preferences.max(dim=-1, keepdim=True)[0]
		

		action_probs = torch.nn.functional.softmax(preferences/self.temperature, dim = -1) + 1e-8
		return action_probs
		
	def update_actor_critic( self, actions_encode: torch.Tensor):

		rewards = torch.tensor([x['reward'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
		state = torch.cat([x['state'] for x in self.buffer]).to(device=self.Actor.device)
		next_state = torch.cat([x['next_state'] for x in self.buffer]).to(device=self.Actor.device)
		action = torch.stack([x['action'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
		dones = torch.tensor([float(x['end_state']) for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)

		masks = torch.cat([x['masked_index'].unsqueeze(0) for x in self.buffer]).to(device=self.Actor.device)

		values = self.Critic(state)
		next_values = self.Critic(next_state).detach()

		# print("rewards", rewards.dtype, "values", values.dtype, "next_values", next_values.dtype, "dones", dones.dtype)
		loss_critic = self.Critic.criterion(rewards + self.gamma*next_values*(1 - dones), values)

		qa = self.policy(state, actions_encode, masks)
		
		log_probs = torch.log(qa.gather(1, action))
		advantage = (rewards + self.gamma * next_values * (1 - dones) - values).detach()

		loss_actor = -torch.mean(log_probs * advantage)

		self.Actor.optimizer.zero_grad()
		self.Critic.optimizer.zero_grad()

		loss_critic.backward()
		loss_actor.backward()

		self.Actor.optimizer.step()
		self.Critic.optimizer.step()

		return loss_critic.item(), loss_actor.item()
	
	def optimization_step(self, actions_encode: torch.Tensor):
		loss_critic, loss_actor = self.update_actor_critic(actions_encode)
		self.buffer = self.buffer[-self.buffer_size + 1:]
		return loss_critic, loss_actor
