import pickle
from matplotlib import pyplot as plt
from glob import glob

from models.SimulationEnvironment import explore_environment
from models.Actor_Critic import AdaptationEngine
from tqdm import tqdm
import torch
import numpy as np
import mlflow

from types import SimpleNamespace


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_entropy(preferences):
	return -torch.sum(preferences * torch.log(preferences), dim = -1)

def train_agent (settings: dict, use_mlflow: bool = True, save_suffix: str = "" ) -> tuple:

    settings = SimpleNamespace(**settings)


    with open("clip_encodings.pkl", "rb") as f:
        data = pickle.load(f)
        image_encodings = data['image_encodings']
        actions_encode = data['concept_encodings']

    with open("generated_concepts.pkl", "rb") as f:
        data = pickle.load(f)
        generated_concepts = data['concepts']
        topic_groups = data['topic_groups']

    with open('encodings.pkl', 'rb') as f:
        captions = pickle.load(f)['captions']

    images = [i for i in glob("../dataset/Images/*.jpg")]
    
    Agent = AdaptationEngine(state_dim = image_encodings.shape[1] + actions_encode.shape[1],
                                action_dim=actions_encode.shape[1],
                                action_space_len=actions_encode.shape[0],
                                lr_actor = settings.lr_actor,
                                lr_critic = settings.lr_critic,
                                gamma = settings.gamma,
                                buffer_size = settings.buffer_size,
                                sample_temperature = settings.temperature,
                                final_temperature = settings.final_temperature,
                                temperature_decay = settings.decay_rate)


    env = explore_environment(topic_groups=topic_groups, 
                            concepts=generated_concepts,
                            concepts_encode=actions_encode,
                            images=images,
                            images_caption=captions,
                            images_encode=image_encodings, 
                            threshold=10)

    #-----------------------------------------------------------------------------

    episode_history = {'rewards': [],
                'loss_critic': [], 
                'loss_actor': []}

    history = {'rewards': [],
            'episodes_rewards': {},
                'loss_critic': [], 
                'loss_actor': []}

    average_reward = [0]
    deviation_reward = [0]

    for episode in range(settings.episodes):

        episode_index, (prev_state, feedback, _) = env.reset()
        prev_state = env.preprocess_state(prev_state)

        env.externalized.add(feedback)

        episode_history = {'rewards': [],
            'loss_critic': [], 
            'loss_actor': [],
            'entropy': []}
        if episode_index not in history['episodes_rewards']:
            history['episodes_rewards'][episode_index] = []
                
        itera = tqdm(range(len(env.topic) - 1))
        itera.set_description(f"Episode {episode}")

        taken_actions = []
        
        for step in itera:
            
            with torch.no_grad():
                Agent.eval()
                preferences = Agent.policy(prev_state, actions_encode)
                # print(prev_state)
                # print(preferences)	
                action = torch.multinomial(preferences, 1).squeeze(-1)
                taken_actions.append(action.item())
                
                state, feedback, reward, done = env.step( action.item() )
                state = env.preprocess_state(state)
                env.externalized.add(feedback)
            # Agent.train()

            Agent.push_buffer(state=prev_state,
                        action=action, 
                        reward=reward, 
                        next_state=state, 
                        done=done, 
                        masked_indices = taken_actions)
            
            episode_history['rewards'].append(reward)
            episode_history['entropy'].append(get_entropy(preferences).mean().item())

            if Agent.buffer_size <= len(Agent.buffer):
                actor_loss, critic_loss = Agent.optimization_step(actions_encode)
                episode_history['loss_critic'].append(critic_loss)
                episode_history['loss_actor'].append(actor_loss)

            prev_state = state

            itera.set_postfix({'Temp': f'{Agent.temperature:.3f}',
                            'Avg. Reward': f"{average_reward[-1]:.3f}",
                            'Actions': len(set(taken_actions)),
                            'Reward': sum(episode_history['rewards']),
                            'entropy': sum(episode_history['entropy']),
                            'L Critic': sum(episode_history['loss_critic']),
                            'L Actor': sum(episode_history['loss_actor'])} )
            if done:
                history['rewards'].append(sum(episode_history['rewards']))
                history['loss_critic'].append(sum(episode_history['loss_critic']))
                history['loss_actor'].append(sum(episode_history['loss_actor']))
                break
        
        history['episodes_rewards'][episode_index].append(sum(episode_history['rewards']).item())

        # average_reward += [np.mean([np.mean(history['episodes_rewards'][i]) for i in history['episodes_rewards'].keys()]) \
        average_reward += [np.mean([history['episodes_rewards'][i][-1] for i in history['episodes_rewards'].keys()]) \
            if len(history['episodes_rewards']) ==  len(env.topic_groups) else -1]
        deviation_reward += [np.std([np.mean(history['episodes_rewards'][i]) for i in history['episodes_rewards'].keys()]) \
        if len(history['episodes_rewards']) ==  len(env.topic_groups) else -1]

        if  Agent.max_reward is not None and Agent.max_reward > -1:
            mlflow.log_metric('average_reward', average_reward[-1], 
                              step = episode)
            mlflow.log_metric('loss_critic', sum(episode_history['loss_critic']), 
                              step = episode)
            mlflow.log_metric('loss_actor', sum(episode_history['loss_actor']), 
                              step = episode)
            mlflow.log_metric('temperature', Agent.temperature, 
                              step = episode)                   

        if Agent.max_reward is None or Agent.max_reward < average_reward[-1]:
            Agent.max_reward = average_reward[-1]
            Agent.Actor.save(f"actor{save_suffix}.pt")
            Agent.Critic.save(f"critic{save_suffix}.pt")
            Agent.temperature = max(Agent.final_temperature, Agent.temperature * Agent.temperature_decay)

            print(f"Model saved - {average_reward[-1]:.2f}")
	
    return average_reward, deviation_reward, history




def plot_exploration(actions_encode, env, taken_actions, generated_concepts):
	"""
	Plot the exploration of two sets of embeddings using t-SNE.
	"""
	# Set random seed for reproducibility
	np.random.seed(42)
	embeddings_1 = actions_encode[env.topic]
	embeddings_2 = actions_encode[taken_actions] 

	# Define set labels
	Set1 = ["Externalized"] * len(embeddings_1)
	Set2 = ["Proposed"] * len(embeddings_2)

	# Define hover labels
	labels_1 = [generated_concepts[i] for i in env.topic]
	labels_2 = [generated_concepts[i] for i in taken_actions]

	# Concatenate embeddings and labels
	embeddings = np.vstack([embeddings_1, embeddings_2])
	labels = labels_1 + labels_2  # Merge labels
	Sets = Set1 + Set2

	# Dimensionality Reduction using TSNE
	reducer = TSNE(n_components=2, perplexity=5, random_state=42)
	reduced_embeddings = reducer.fit_transform(embeddings)

	# Extract x, y coordinates
	x, y = reduced_embeddings[:, 0], reduced_embeddings[:, 1]

	# Create Matplotlib figure
	plt.figure(figsize=(10, 7))

	# Plot each set separately with different colors
	for set_type, color in zip(["Externalized", "Proposed"], ["red", "blue"]):
		idx = [i for i in range(len(Sets)) if Sets[i] == set_type]
		plt.scatter(x[idx], y[idx], label=set_type, color=color, alpha=0.7)

	# Add labels next to each point
	for i in range(len(labels)):
		plt.text(x[i], y[i], labels[i], fontsize=8, ha='right', va='bottom', alpha=0.8)
	
	# Customize plot
	plt.xlabel("t-SNE Component 1")
	plt.ylabel("t-SNE Component 2")
	plt.title("Comparison of Two Embedding Sets")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.6)	

	# Show plot
	plt.show()

def get_trajectory( env_snapshot, state, trajectory_len: int = 10 ):
	
	current_path = []
	processed_state = env_snapshot.preprocess_state(state)

	for _ in range(trajectory_len):
		
		preferences = Agent.policy(processed_state, actions_encode)
		action = preferences.argmax() #torch.multinomial(preferences, 1).squeeze(-1)
		
		current_path.append(action.item())
		processed_state = env_snapshot.preprocess_state(state, current_path=current_path)
	
	return current_path

def simulate(Agent, topic_groups, generated_concepts, actions_encode, image_encodings, captions, images):


	Agent.eval()
	env_tmp = explore_environment(topic_groups=topic_groups, 
							concepts=generated_concepts,
							concepts_encode=actions_encode,
							images=images,
							images_caption=captions,
							images_encode=image_encodings, 
							threshold=10)

	taken_actions = []

	with torch.no_grad():
		for episode in range(1):

			episode_index, (prev_state, feedback, _) = env_tmp.reset()
			prev_state = env_tmp.preprocess_state(prev_state)

			env_tmp.externalized.add(feedback)
					
			itera = tqdm(range(len(env_tmp.topic)))
			itera.set_description(f"Episode {episode}")

			for step in itera:

				# actions_trajectory = get_trajectory(env, prev_state, trajectory_len = 10)

				# prediction.append(actions_trajectory.copy())
				# remaining.append(env.topic[env.step_index:].copy())
				preferences = Agent.policy(prev_state, actions_encode)
				action = torch.multinomial(preferences, 1).squeeze(-1)

				state, feedback, _, done = env_tmp.step( action.item() )
				taken_actions.append(action.item())
				print(action.item(), generated_concepts[action.item()])
				state = env_tmp.preprocess_state(state)
				# state, feedback, _, done = env.step( actions_trajectory[0] )
				env_tmp.externalized.add(feedback)
				# actions_trajectory = state
				prev_state = state

				if done:
					break		

	plot_exploration(actions_encode, env_tmp, taken_actions, generated_concepts)
