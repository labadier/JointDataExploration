import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)


def generate_episode(topic_groups, concept_instance, images, threshold = 10):

	"""
    Generates an episode consisting of batches of images explored by the user before introducing
	a concept, ensuring that concepts appear at least a specified number of times within the batches.

    Parameters:
    -----------
    topic_groups : dict
        A dictionary where keys represent topic labels and values are lists of associated concepts.
    concept_instance : dict
        A dictionary mapping each concept to a set of images representing that concept.
    images : list
        A list of all available images.
    threshold : int, optional
        The minimum number of times each concept should appear in the episode (default is 10).

    Returns:
    --------
    images_batches : list of sets
        A list where each element is a batch (set) of images corresponding to a step in the episode.
	topic : list
		The list of externalized concepts in the selected topic.
    """
	
	images_batches = []
	observed_images = set()
	threshold = 10

	topic = topic_groups[np.random.choice(list(topic_groups.keys()))]
	random.shuffle(topic)
	concept_appereance = {i:set() for i in topic}

	for step in topic:

		if len(concept_appereance[step]) < threshold:
			sample1 = set(random.sample(concept_instance[step], 
										threshold - len(concept_appereance[step])))
			
			sample = sample1 | set(random.sample(images, 
										len(images)//len(topic) - len(sample1))) 
		else:
			sample = set(random.sample(images, len(images)//len(topic))) 

		observed_images |= sample
		images_batches.append(sample)

		for key in concept_appereance:

			added = set(concept_instance[key]) & sample
			concept_appereance[key] |= added

	return images_batches, topic




#actor critic model
class Actor(torch.nn.Module):
	
    
	def __init__(self, state_dim: int, action_dim: int,
			  lr_optimizer: float = 0.01):
		super(Actor, self).__init__()

		self.model = torch.nn.Sequential(
			torch.nn.Linear(state_dim, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, action_dim),
		)

		self.criterion = torch.nn.MSELoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_optimizer)
		
	def forward(self, state):
		return self.model(state.to(self.device))
		# return self.model(torch.cat([state, action], dim = -1).to(self.device)).squeeze(-1)
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path, map_location=self.device))
    
	

class Critic(torch.nn.Module):

	def __init__(self, state_dim: int,
			  lr_optimizer: float = 0.01):
		super(Critic, self).__init__()
		
		self.model = torch.nn.Sequential(
			torch.nn.Linear(state_dim, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 1),
		)

		self.criterion = torch.nn.MSELoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_optimizer)
		

	def forward(self, state):
		return self.model(state.to(self.device))

	def save(self, path):
		torch.save(self.state_dict(), path)


class AdaptationEngine(torch.nn.Module):

	def __init__(self, state_dim: int, 
					action_dim: int,
					lr_actor: float = 0.001,
					lr_critic: float = 0.0001,
					gamma: float = 0.99,
					buffer_size: int = 512,
					sample_temperature: float = 5.0,
					final_temperature: float = 0.1,
					temperatura_decay: float = 0.99,
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
		self.temperature_decay = temperatura_decay

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.max_reward = None


	def push_buffer(self, state, action, reward, next_state, done):
		self.buffer.append({'state': state,
							'action': action, 
							'reward': reward, 
							'next_state': next_state, 
							'end_state': done})
		
	def policy(self, state, actions_encode):
	
		action_latent = self.Actor.forward(state) #b x action_dim
		preferences = action_latent @ actions_encode.T
		preferences -= preferences.max(dim=-1, keepdim=True)[0]

		action_probs = torch.nn.functional.softmax(preferences/self.temperature, dim = -1) + 1e-8
		return action_probs
		
	def update_actor_critic( self, actions_encode: torch.Tensor):

		rewards = torch.tensor([x['reward'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
		state = torch.cat([x['state'] for x in self.buffer]).to(device=self.Actor.device)
		next_state = torch.cat([x['next_state'] for x in self.buffer]).to(device=self.Actor.device)
		action = torch.stack([x['action'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
		dones = torch.tensor([float(x['end_state']) for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)


		values = self.Critic(state)
		next_values = self.Critic(next_state).detach()

		loss_critic = self.Critic.criterion(rewards + self.gamma*next_values*(1 - dones), values)

		qa = self.policy(state, actions_encode)
		
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
