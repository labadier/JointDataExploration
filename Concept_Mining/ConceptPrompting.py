import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

from sklearn.metrics.pairwise import cosine_similarity

from glob import glob

class explore_environment:

	def __init__(self, topic_groups, 
			  concepts, 
			  concepts_encode,
			  images, 
			  images_caption,
			  images_encode,
			  threshold = 10,
			  reward_lambda = 0.5):
		
		self.topic_groups = topic_groups
		self.concepts = concepts

		self.images = list(range(len(images)))#images
		self.images_encode = images_encode
		self.threshold = threshold

		self.images_batches = None
		self.topic = None
		self.concept_instance = None
		self.preprocess_topic_groups(images_caption)
		self.step_index = 0

		self.externalized = set()
		self.obsrved_images = set()

		self.action_encode = concepts_encode
		self.reward_lambda = reward_lambda

		self.taken_actions = []
		self.diversity_threshold = 0.5

	def preprocess_topic_groups(self, images_caption):

		self.concept_instance = {i:[] for i in self.concepts}

		for i, path in enumerate(self.images):
			for c in images_caption[i].split():
				if c in self.concepts:
					self.concept_instance[c] += [path]

		remove_topics = []
		for i in self.topic_groups:
			remove = [j for j in self.topic_groups[i] if len(self.concept_instance[j]) < self.threshold]
			for j in remove:
				self.topic_groups[i].remove(j)

			if len(self.topic_groups[i]) < 8:
				remove_topics.append(i)

		for i in remove_topics:
			del self.topic_groups[i]
		for i in self.topic_groups:
			print(f"Topic {i}:", len(self.topic_groups[i]))

	def reset(self):
		self.images_batches, self.topic, topic_index = self.generate_episode()
		self.step_index = 0
		self.obsrved_images = set()
		self.externalized = set()
		self.taken_actions = []
		return topic_index, self.step()
	

	def step(self, action = None ):

		if self.step_index < len(self.images_batches):
			images = self.images_batches[self.step_index]
			topics = self.topic[self.step_index]
			self.step_index += 1

			if action is None:
				return images, topics, False

			# compute reward
			observation_memmorized = self.images_encode[list(self.obsrved_images)] #.mean(dim=0).unsqueeze(0) if self.obsrved_images else torch.zeros_like(new_observation)
			new_observation = self.images_encode[list(images)]#.mean(dim=0).unsqueeze(0)
			
			externalized_concepts = self.action_encode[list(self.externalized)] #.mean(dim=0).unsqueeze(0)

			actual_action = self.action_encode[action].unsqueeze(0)

			observation_compatibility = cosine_similarity(new_observation, actual_action).mean()
			history_compatibility = cosine_similarity(externalized_concepts, actual_action).mean()
			dataset_compatibility = cosine_similarity(observation_memmorized, actual_action).mean()
			# current_action_compability = cosine_similarity(actual_action, actual_action).mean()

			current_batch_similarity = cosine_similarity(actual_action, self.action_encode[self.taken_actions]).mean() if self.taken_actions else 0
			diversity_penalty = min(0, self.diversity_threshold - current_batch_similarity)


			# reward = self.reward_lambda/2 * actual_action @ new_observation.T \
			# 	+ self.reward_lambda/2 *actual_action @ observation_memmorized.T \
			# 	+ (1-self.reward_lambda) * actual_action @ externalized_concepts.T

			reward = self.reward_lambda/2 * observation_compatibility \
				+ self.reward_lambda/2 * dataset_compatibility \
				+ (1-self.reward_lambda) * history_compatibility\
				+ current_batch_similarity \
				+ diversity_penalty
			# print(type(reward), reward)
			return images, topics, np.float32(reward), False
		

		else:
			return [], -1, 0, True 
		
		
	def generate_episode(self):
		
		images_batches = []
		observed_images = set()
		threshold = 10

		topic_index = np.random.choice(list(self.topic_groups.keys()))
		topic = self.topic_groups[topic_index]
		random.shuffle(topic)
		concept_appereance = {i:set() for i in topic}

		for step in topic:

			if len(concept_appereance[step]) < threshold:
				sample1 = set(random.sample(self.concept_instance[step], 
											threshold - len(concept_appereance[step])))
				
				sample = sample1 | set(random.sample(self.images, 
											len(self.images)//len(topic) - len(sample1))) 
			else:
				sample = set(random.sample(self.images, len(self.images)//len(topic))) 

			observed_images |= sample
			images_batches.append(sample)

			for key in concept_appereance:

				added = set(self.concept_instance[key]) & sample
				concept_appereance[key] |= added

		topic = [self.concepts.index(i) for i in topic]
		return images_batches, topic, topic_index
	
	def preprocess_state(self, images):
		 
		new_observation = self.images_encode[list(images)].mean(dim=0).unsqueeze(0) if images else torch.zeros(1, self.images_encode.shape[-1])
		observation_memmorized = self.images_encode[list(self.obsrved_images)].mean(dim=0).unsqueeze(0) if self.obsrved_images else torch.zeros_like(new_observation)

		externalized_concepts = self.action_encode[list(self.externalized)].mean(dim=0).unsqueeze(0) if self.obsrved_images else torch.zeros_like(new_observation)
		observation = torch.cat([new_observation + observation_memmorized, externalized_concepts], dim=-1)

		self.obsrved_images |= set(images)
		
		return observation


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
		preferences = action_latent @ actions_encode.to(action_latent.device).T
		
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


		values = self.Critic(state)
		next_values = self.Critic(next_state).detach()

		# print("rewards", rewards.dtype, "values", values.dtype, "next_values", next_values.dtype, "dones", dones.dtype)
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
