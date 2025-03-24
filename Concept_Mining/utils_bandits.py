import os
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

import pickle
from matplotlib import pyplot as plt
from glob import glob

from SimulationEnvironment import explore_environment
from ContextualBandit import AdaptationEngine
from tqdm import tqdm
from sklearn.manifold import TSNE

import mlflow

from types import SimpleNamespace

def get_entropy(preferences):
	return -torch.sum(preferences * torch.log(preferences), dim = -1)

def train_agent (settings: dict, use_mlflow: bool = True, 
                 output_path: str = '.', save_suffix: str = "",
                 use_rnn: bool = False) -> tuple:

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
                                gamma = settings.gamma,
                                buffer_size = settings.buffer_size,
                                sample_temperature = settings.temperature,
                                final_temperature = settings.final_temperature,
                                temperature_decay = settings.decay_rate,
                                use_rnn = use_rnn)


    env = explore_environment(topic_groups=topic_groups, 
                            concepts=generated_concepts,
                            concepts_encode=actions_encode,
                            images=images,
                            images_caption=captions,
                            images_encode=image_encodings, 
                            threshold=10)

    #-----------------------------------------------------------------------------

    episode_history = {'rewards': [],
                'loss_actor': []}

    history = {'rewards': [],
            'episodes_rewards': {},
                'loss_actor': []}

    average_reward = [0]
    deviation_reward = [0]

    for episode in range(settings.episodes):

        episode_index, (prev_state, feedback, _) = env.reset()
        prev_state, prev_action_seq = env.preprocess_state(prev_state)

        env.externalized.add(feedback)

        episode_history = {'rewards': [],
            'loss_actor': [],
            'entropy': []}
        if episode_index not in history['episodes_rewards']:
            history['episodes_rewards'][episode_index] = []
                
        itera = tqdm(range(len(env.topic) - 1))
        itera.set_description(f"Episode {episode}")

        taken_actions = []
        
        for _ in itera:
            
            with torch.no_grad():
                Agent.eval()
                if use_rnn:
                    preferences = Agent.policy(prev_state, actions_encode, prev_action_seq)
                else:
                    preferences = Agent.policy(prev_state, actions_encode)	
                    
                
                if random.random() < Agent.temperature:
                    action = torch.randint(0, len(env.action_encode), (1,)).item()
                else:
                    action = torch.argmax(preferences).item()

                taken_actions.append(action)
                
                state, feedback, reward, done = env.step( action )
                state, action_seq = env.preprocess_state(state)

                # done |= len(set(env.topic[env.step_index+1:]) - set(taken_actions))

                if feedback in taken_actions:
                    continue   

                env.externalized.add(feedback)
            Agent.train()

            Agent.push_buffer(state=prev_state,
                            action_sequence=prev_action_seq,
                            action=torch.tensor(action), 
                            reward=reward, 
                            next_state=state, 
                            next_action_seq=action_seq,
                            done=done, 
                            masked_indices = taken_actions)
            
            episode_history['rewards'].append(reward)
            episode_history['entropy'].append(get_entropy(preferences).mean().item())

            if Agent.buffer_size <= len(Agent.buffer):
                actor_loss = Agent.optimization_step(actions_encode)
                episode_history['loss_actor'].append(actor_loss)

            prev_state = state
            prev_action_seq = action_seq

            itera.set_postfix({'Temp': f'{Agent.temperature:.3f}',
                            'Avg. Reward': f"{average_reward[-1]:.3f}",
                            'Actions': len(set(taken_actions)),
                            'Reward': sum(episode_history['rewards']),
                            'entropy': sum(episode_history['entropy']),
                            'L Actor': sum(episode_history['loss_actor'])} )
            if done:
                history['rewards'].append(sum(episode_history['rewards']))
                history['loss_actor'].append(sum(episode_history['loss_actor']))
                break
        
        history['episodes_rewards'][episode_index].append(sum(episode_history['rewards']).item())

        # average_reward += [np.mean([np.mean(history['episodes_rewards'][i]) for i in history['episodes_rewards'].keys()]) \
        average_reward += [np.mean([history['episodes_rewards'][i][-1] for i in history['episodes_rewards'].keys()]) \
            if len(history['episodes_rewards']) ==  len(env.topic_groups) else -1]
        deviation_reward += [np.std([np.mean(history['episodes_rewards'][i]) for i in history['episodes_rewards'].keys()]) \
        if len(history['episodes_rewards']) ==  len(env.topic_groups) else -1]

        if use_mlflow and Agent.max_reward is not None and Agent.max_reward > -1:
            mlflow.log_metric('average_reward', average_reward[-1], 
                                step = episode)
            mlflow.log_metric('loss_bandit', sum(episode_history['loss_actor']), 
                                step = episode)
            mlflow.log_metric('temperature', Agent.temperature, 
                                step = episode)                   

        if Agent.max_reward is None or Agent.max_reward < average_reward[-1]:
            Agent.max_reward = average_reward[-1]
            Agent.Actor.save(os.path.join(output_path, f"bandit{save_suffix}.pt"))
            Agent.temperature = max(Agent.final_temperature, Agent.temperature * Agent.temperature_decay)

            print(f"Model saved - {average_reward[-1]:.2f}")

        if episode % 1000 == 0 or episode == settings.episodes - 1:
            os.makedirs(os.path.join(output_path, str(episode)), exist_ok=True)
            for i in env.topic_groups.keys():
                simulate(Agent=Agent, topic_groups=topic_groups,
                        generated_concepts=generated_concepts, actions_encode=actions_encode,
                            image_encodings=image_encodings, captions=captions, images=images, 
                            topic_index=i, save_plot = True, 
                            output_path = os.path.join(output_path, str(episode)))

    Agent.Actor.load(os.path.join(output_path, f"bandit{save_suffix}.pt"))
    os.makedirs(os.path.join(output_path, "best"), exist_ok=True)
    for i in env.topic_groups.keys():
        simulate(Agent=Agent, topic_groups=topic_groups,
                generated_concepts=generated_concepts, actions_encode=actions_encode,
                image_encodings=image_encodings, captions=captions, images=images, 
                topic_index=i, save_plot = True, 
                output_path = os.path.join(output_path, "best")) 

    return average_reward, deviation_reward, history


def plot_exploration(actions_encode, env, taken_actions, generated_concepts,
                    save_plot: bool = False, episode_index: int = 0, output_path: str = "."):
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
    if save_plot:
        plt.savefig(os.path.join(output_path, f"e_{episode_index}.png"))
        plt.close()
        with open(os.path.join(output_path,f"e_{episode_index}.txt"), "a") as f:
             f.write('Ground Truth: ' + ','.join(labels_1) + '\n')
             f.write('Proposed: ' + ','.join(labels_2) + '\n')             
    else:
        plt.show()
        print('Ground Truth: ' + ','.join(labels_1) + '\n')
        print('Proposed: ' + ','.join(labels_2) + '\n')   

def get_trajectory( Agent, actions_encode, env_snapshot, state, trajectory_len: int = 10 ):
	
	current_path = []
	processed_state = env_snapshot.preprocess_state(state)

	for _ in range(trajectory_len):
		
		preferences = Agent.policy(processed_state, actions_encode)
		action = preferences.argmax() #torch.multinomial(preferences, 1).squeeze(-1)
		
		current_path.append(action.item())
		processed_state = env_snapshot.preprocess_state(state, current_path=current_path)
	
	return current_path

def simulate(Agent, topic_groups, generated_concepts, actions_encode, 
             image_encodings, captions, images, topic_index = None,
             save_plot: bool = False, output_path: str = "."):


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

            episode_index, (prev_state, feedback, _) = env_tmp.reset(topic_index)
            prev_state, prev_action_seq = env_tmp.preprocess_state(prev_state)

            env_tmp.externalized.add(feedback)
            
            if not save_plot:
                itera = tqdm(range(len(env_tmp.topic)))
                itera.set_description(f"Episode {episode}")
            else:
                itera = range(len(env_tmp.topic))

            for step in itera:

                # actions_trajectory = get_trajectory(env, prev_state, trajectory_len = 10)
                preferences = Agent.policy(prev_state, actions_encode, prev_action_seq if Agent.use_rnn else None).cpu()[0]
                preferences[taken_actions] = -1<<32
                action = preferences.argmax() #torch.multinomial(preferences, 1).squeeze(-1)

                state, feedback, _, done = env_tmp.step( action )
                taken_actions.append(action.item())
                # print(action.item(), generated_concepts[action.item()])
                state, action_seq = env_tmp.preprocess_state(state)
                # state, feedback, _, done = env.step( actions_trajectory[0] )
                env_tmp.externalized.add(feedback)
                # actions_trajectory = state
                prev_state = state
                prev_action_seq = action_seq

                if done:
                    break		

    plot_exploration(actions_encode = actions_encode, env = env_tmp, 
                taken_actions = taken_actions, generated_concepts = generated_concepts, 
                save_plot = save_plot, episode_index = topic_index,
                output_path = output_path)