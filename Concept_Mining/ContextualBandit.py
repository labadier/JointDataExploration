import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

from sklearn.metrics.pairwise import cosine_similarity

from glob import glob

class Actor(torch.nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int,
            lr_optimizer: float = 0.01, use_rnn: bool = False):
        super(Actor, self).__init__()

        
        if use_rnn:
            self.text_gru = torch.nn.GRU(
                input_size=action_dim,
                hidden_size=64,
                batch_first=True
            )
            self.text_proj = torch.nn.Linear(64, 256)
        else:
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
            # torch.nn.Linear(state_dim, 128),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

        self.use_rnn = use_rnn
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_optimizer)
        
    def forward(self, state, action_seq = None):
        
        state = state.to(self.device)
        i = self.condense_image(state[..., :state.shape[-1] >> 1])
        if action_seq is None:
            t = self.condense_text(state[..., state.shape[-1] >> 1:])
        else:
            masks = action_seq[1]
            action_seq = action_seq[0].to(self.device)
            hidden, _ = self.text_gru(action_seq) # hidden -> b, len(action_seq), 64
            # take the relevant hidden state acording to the mask
            last_nonzero_idx = torch.argmax(masks.cumsum(1), dim=1)
            
            # FIX THIS COCHINADA
            logits = torch.stack([hidden[i][non_zero] for i, non_zero in enumerate(last_nonzero_idx.tolist())]) # b, 64
            t = self.text_proj(logits)

        return self.head(i + t)
        # return self.head(state)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
    
class AdaptationEngine(torch.nn.Module):

    def __init__(self, state_dim: int, 
                    action_dim: int,
                    action_space_len: int,
                    lr_actor: float = 0.001,
                    gamma: float = 0.99,
                    buffer_size: int = 512,
                    sample_temperature: float = 5.0,
                    final_temperature: float = 0.1,
                    temperature_decay: float = 0.99,
                    use_rnn: bool = False,
                    **kwargs
                    ):
        super(AdaptationEngine, self).__init__()

        self.Actor = Actor(state_dim = state_dim, action_dim = action_dim,
                            lr_optimizer = lr_actor, use_rnn = use_rnn)
        
        self.gamma = gamma
        self.use_rnn = use_rnn
        self.buffer_size = buffer_size
        self.buffer = []
        self.temperature = sample_temperature
        self.final_temperature = final_temperature
        self.temperature_decay = temperature_decay
        self.action_space_len = action_space_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_reward = None


    def push_buffer(self, state, action_sequence, action, reward, next_state,
                     next_action_seq, done, masked_indices = None):

        mask = torch.zeros(self.action_space_len)
        mask[masked_indices] = 1

        self.buffer.append({'state': state,
                            'action_seq': action_sequence,
                            'action': action, 
                            'reward': reward, 
                            'next_state': next_state,
                            'next_action_seq': next_action_seq,
                            'end_state': done,
                            'masked_index': mask})
        
    def policy(self, state, actions_encode, action_seq = None):
        
        action_latent = self.Actor.forward(state, action_seq=action_seq) #b xaction_dim
        preferences = action_latent @ actions_encode.to(action_latent.device).T
        
        # if masking_actions is not None:
        #     preferences = preferences +  -1e8*masking_actions.to(preferences.device)
            
        return preferences
        
    def update_actor_critic( self, actions_encode: torch.Tensor):

        rewards = torch.tensor([x['reward'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
        state = torch.cat([x['state'] for x in self.buffer]).to(device=self.Actor.device)
        next_state = torch.cat([x['next_state'] for x in self.buffer]).to(device=self.Actor.device)
        action = torch.stack([x['action'] for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
        # seq modelling
        if self.use_rnn:
            action_seq = torch.cat([x['action_seq'][0] for x in self.buffer]).to(device=self.Actor.device)
            seq_masks = torch.cat([x['action_seq'][1] for x in self.buffer]).to(device=self.Actor.device)

            next_action_seq = torch.cat([x['next_action_seq'][0] for x in self.buffer]).to(device=self.Actor.device)
            next_seq_masks = torch.cat([x['next_action_seq'][1] for x in self.buffer]).to(device=self.Actor.device)
        #
        dones = torch.tensor([float(x['end_state']) for x in self.buffer]).to(device=self.Actor.device).reshape(-1, 1)
        masks = torch.cat([x['masked_index'].unsqueeze(0) for x in self.buffer]).to(device=self.Actor.device)

        qa = self.policy(state, actions_encode, (action_seq, seq_masks) if self.use_rnn else None)
        qa_star = self.policy(next_state, actions_encode, (next_action_seq, next_seq_masks) if self.use_rnn else None)
        
        #the best action in the next state
        q_start_values = torch.max(qa_star, 1)[0].unsqueeze(1).detach()
        # print(qa, q_start_values)
        # print(state)
        # print(fwef)
        
        loss_actor = self.Actor.criterion(rewards.to(self.device) + self.gamma*dones*q_start_values,
                                qa.gather(1, action) ).mean()

        self.Actor.optimizer.zero_grad()
        loss_actor.backward()

        self.Actor.optimizer.step()

        return loss_actor.item()
    
    def optimization_step(self, actions_encode: torch.Tensor):
        loss_actor = self.update_actor_critic(actions_encode)
        self.buffer = self.buffer[-self.buffer_size + 1:]
        return loss_actor
