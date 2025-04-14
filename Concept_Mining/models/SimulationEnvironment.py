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
		# for i in self.topic_groups:
		# 	print(f"Topic {i}:", len(self.topic_groups[i]))

	def reset(self, index = None):
		self.images_batches, self.topic, topic_index = self.generate_episode(index)
		self.step_index = 0
		self.obsrved_images = set()
		self.externalized = set()
		self.taken_actions = []
		return topic_index, self.step()
	

	def step(self, action = None ):
		
		end_episode = False
		if self.step_index < len(self.images_batches) - 1:
			self.step_index += 1
			images = self.images_batches[self.step_index]
			concept = self.topic[self.step_index - 1]

			if action is None:
				return images, concept, False
		else: 
			images = []
			concept = self.topic[self.step_index]
			end_episode = True
			
		
		future = self.topic[self.step_index - 1:]
		# past = self.topic[:self.step_index - 1]
		
		if len(future) == 0:
			future_concepts = self.action_encode[list(self.topic[self.step_index-1:])].unsqueeze(0) #.mean(dim=0).unsqueeze(0)
		else:
			future_concepts = self.action_encode[future]

		actual_action = self.action_encode[action].unsqueeze(0)
		future_similarity = cosine_similarity(actual_action, future_concepts)

		if action in future:
			#swap self.topic[self.step_index] with the position in the future to make the episode dynamic
			a = self.step_index - 1
			b = self.topic.index(action)
			# print(a, b)
			self.topic[a], self.topic[b] = self.topic[b], self.topic[a]
			concept = self.topic[a]
			reward = 5
		elif action in self.externalized or future_similarity.max() < 0.97:
			reward = -5
		elif future_similarity.max() > 0.97:
			reward = 5
		
		# action in self.taken_actions:
		# 	reward = future_similarity.max() if action not in self.topic[self.step_index-1:] else 2
		
		return images, concept, np.float32(reward), end_episode, (action in future)
		
		
		
	def generate_episode(self, index = None):
		
		images_batches = []
		observed_images = set()
		threshold = 10

		topic_index = np.random.choice(list(self.topic_groups.keys())) if index is None else index
		topic = self.topic_groups[topic_index]
		# random.shuffle(topic)
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
	
	def preprocess_state(self, images, current_path = [], max_path_len = 30):
		 
		new_observation = self.images_encode[list(images)].mean(dim=0).unsqueeze(0) if images else torch.zeros(1, self.images_encode.shape[-1])
		# observation_memmorized = self.images_encode[list(self.obsrved_images)].mean(dim=0).unsqueeze(0) if self.obsrved_images else torch.zeros_like(new_observation)

		externalized_concepts = self.action_encode[list(self.externalized) + current_path].mean(dim=0).unsqueeze(0) if self.obsrved_images else torch.zeros_like(new_observation)
		
		action_sequence = self.action_encode[list(self.externalized)[-max_path_len:]]
		active_tokens = torch.ones(action_sequence.shape[0])
		if action_sequence.shape[0] < max_path_len:
			action_sequence = torch.cat([torch.zeros(max_path_len - action_sequence.shape[0], action_sequence.shape[-1]), action_sequence], dim=0)
			active_tokens = torch.cat([active_tokens, torch.zeros(max_path_len - active_tokens.shape[0])], dim=0)

		observation = torch.cat([new_observation, externalized_concepts], dim=-1)
		# observation = torch.cat([(new_observation + observation_memmorized)/2.0, externalized_concepts], dim=-1)

		self.obsrved_images |= set(images)
		
		return observation, (action_sequence.unsqueeze(0) , active_tokens.unsqueeze(0) )
