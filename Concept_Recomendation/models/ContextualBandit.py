import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)

from sklearn.metrics.pairwise import cosine_similarity

from glob import glob

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class Actor(torch.nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int,
            lr_optimizer: float = 0.01, use_rnn: bool = False):
        super(Actor, self).__init__()

        
        if use_rnn:
            self.text_gru = torch.nn.GRU(
                input_size=action_dim,
                hidden_size=128,
                batch_first=True,
                num_layers=3,
            )
            self.text_proj = torch.nn.Linear(128, 256)

            self.condense_image = torch.nn.Sequential(
                torch.nn.Linear(state_dim >> 1, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128))
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
        
        # self.head = torch.nn.Sequential(
        #     # torch.nn.Linear(state_dim, 128),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128), 
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, action_dim),
        # )

        self.head = torch.nn.Sequential(
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
            # print(i.shape)
            hidden, _ = self.text_gru(action_seq, i.unsqueeze(0).repeat([3, 1, 1])) # hidden -> b, len(action_seq), 64
            # take the relevant hidden state acording to the mask
            last_nonzero_idx = torch.argmax(masks.cumsum(1), dim=1)
            
            # FIX THIS COCHINADA
            logits = torch.stack([hidden[i][non_zero] for i, non_zero in enumerate(last_nonzero_idx.tolist())]) # b, 64
            t = self.text_proj(logits)

        return self.head(t)
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

    def action_selection_heuristic(self, vision_state, action_seq, taken_actions, actions_encode):

        # the heuristic will be the action that is the most similar to the average of the action sequence
        avg_action_seq = action_seq[0].sum(dim=1)/action_seq[1].sum(dim=1, keepdim=True) # b x action_dim @ action_dim x action_space_len
        # print(avg_action_seq.shape)


        heuristic = avg_action_seq @ actions_encode.to(avg_action_seq.device).T # b x action_space_len
        assert taken_actions.shape == heuristic.shape and torch.all(torch.logical_or(taken_actions == 0, taken_actions == 1))
        
        # mask taken actions

        compatibility_image = vision_state @ actions_encode.to(vision_state.device).T # b x action_space_len
        heuristic = heuristic*0.5 + compatibility_image*0.5


        heuristic[taken_actions == 1] = float('-inf')


        _, top_k_indices = torch.topk(heuristic, 50, dim=1, largest=True)# preserve only the top 40
        mask = torch.zeros_like(heuristic, dtype=torch.bool)
        mask.scatter_(1, top_k_indices, True)
        heuristic[~mask] = float('-inf')  # or tensor[~mask] = 0
        
        # if vision_state.shape[0] > 1:
        #     print(heuristic.max(), heuristic.min())

        return heuristic
        

        
    def policy(self, state, actions_encode, 
               taken_actions=None, action_seq = None, 
               use_heuristic = True):

        
        assert (action_seq is None and taken_actions is None) or (action_seq is not None and taken_actions is not None), "action_seq and taken_actions must be both None or both not None"
        
        action_latent = self.Actor.forward(state, action_seq=action_seq) #b xaction_dim
        preferences = action_latent @ actions_encode.to(action_latent.device).T

        if taken_actions is None:
            return preferences
        
        if type(taken_actions) == list:
            assert preferences.shape[0] == 1

            one_hot = torch.zeros_like(preferences, dtype=torch.long).to(preferences.device)
            one_hot[0, taken_actions] = 1
            taken_actions = one_hot
            
        assert preferences.shape == taken_actions.shape, f"preferences shape: {preferences.shape}, taken_actions shape: {taken_actions.shape}"
        
        
        if use_heuristic:
            heuristic = self.action_selection_heuristic(state[..., :state.shape[-1] >> 1], action_seq, taken_actions, actions_encode)
            preferences = preferences + heuristic.to(preferences.device)

            # if preferences.shape[0] > 1:
            #     print(heuristic.max(), heuristic.min(), preferences.max(), preferences.min())
            # print('using heuristic')
        else: 
            preferences += -1e8*taken_actions

        return preferences
        
    def update_agent( self, actions_encode: torch.Tensor, use_heuristic: bool = True):

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


        qa = self.policy(state = state, 
                         actions_encode = actions_encode,
                         taken_actions = masks if self.use_rnn else None,
                         action_seq = (action_seq, seq_masks) if self.use_rnn else None,
                         use_heuristic = False)
        
        qa_star = self.policy(state = next_state, 
                              actions_encode = actions_encode, 
                              taken_actions = masks if self.use_rnn else None,
                              action_seq = (next_action_seq, next_seq_masks) if self.use_rnn else None,
                              use_heuristic = False)
        # print(qa.gather(1, action))
        #the best action in the next state
        q_start_values = torch.max(qa_star, 1)[0].unsqueeze(1).detach()
        # print(q_start_values.max(), q_start_values.min())
        # print(state)
        loss_actor = self.Actor.criterion(rewards.to(self.device) + self.gamma*dones*q_start_values,
                                qa.gather(1, action) ).mean()

        self.Actor.optimizer.zero_grad()
        loss_actor.backward()

        self.Actor.optimizer.step()

        # print(f"{bcolors.OKGREEN}Updated Agent!!{bcolors.ENDC}")
        return loss_actor.item()
    
    def optimization_step(self, actions_encode: torch.Tensor, use_heuristic: bool = True):
        loss_actor = self.update_agent(actions_encode, use_heuristic=use_heuristic)
        self.buffer = self.buffer[-self.buffer_size + 1:]
        return loss_actor
