import pickle
from matplotlib import pyplot as plt
from glob import glob

from ConceptPrompting import AdaptationEngine, explore_environment
from tqdm import tqdm
import torch
import numpy as np
import mlflow

from types import SimpleNamespace

def get_entropy(preferences):
	return -torch.sum(preferences * torch.log(preferences), dim = -1)

def train_agent (settings: dict ) -> tuple:

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

    start_logging = False

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

        
        for step in itera:

            preferences = Agent.policy(prev_state, actions_encode)
            # print(prev_state)
            # print(preferences)	
            action = torch.multinomial(preferences, 1).squeeze(-1)
            
            state, feedback, reward, done = env.step( action.item() )
            state = env.preprocess_state(state)
            env.externalized.add(feedback)

            Agent.push_buffer(state=prev_state,
                        action=action, 
                        reward=reward, 
                        next_state=state, 
                        done=done)
            
            episode_history['rewards'].append(reward)
            episode_history['entropy'].append(get_entropy(preferences).mean().item())

            if Agent.buffer_size <= len(Agent.buffer):
                actor_loss, critic_loss = Agent.optimization_step(actions_encode)
                episode_history['loss_critic'].append(critic_loss)
                episode_history['loss_actor'].append(actor_loss)

            prev_state = state

            itera.set_postfix({'Temp': f'{Agent.temperature:.4f}',
                            'Avg. Reward': f"{average_reward[-1]:.4f}",
                            'Reward': sum(episode_history['rewards']),
                            'entropy': sum(episode_history['entropy']),
                            'L Critic': sum(episode_history['loss_critic']),
                            'L Actor': sum(episode_history['loss_actor'])} )
            if done:
                history['rewards'].append(sum(episode_history['rewards']))
                history['loss_critic'].append(sum(episode_history['loss_critic']))
                history['loss_actor'].append(sum(episode_history['loss_actor']))
                break

        Agent.temperature = max(Agent.final_temperature, Agent.temperature * Agent.temperature_decay)

        average_reward += [np.mean([np.mean(history['episodes_rewards'][i]) for i in history['episodes_rewards'].keys()]) \
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
            start_logging |= True

            Agent.max_reward = average_reward[-1]
            Agent.Actor.save(f"actor.pt")
            Agent.Critic.save(f"critic.pt")
            print(f"Model saved - {average_reward[-1]:.2f}")
        history['episodes_rewards'][episode_index].append(sum(episode_history['rewards']).item())

    return average_reward, deviation_reward, history