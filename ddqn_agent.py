import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from per import PrioritizedReplayBuffer


BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 10000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step + 1) 
        self.gamma = 0.99

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) >= self.n_step:
            state, action, reward, next_state, done_sample = self._get_n_step_info()
            self.memory.add(1.0, (state, action, reward, next_state, done_sample))
        
        if done:
            while len(self.n_step_buffer) > 0:
                state, action, reward, next_state, done_sample = self._get_n_step_info()
                self.memory.add(1.0, (state, action, reward, next_state, done_sample))

    def _get_n_step_info(self):
        state, action = self.n_step_buffer[0][:2]
        
        reward = 0
        for i in range(len(self.n_step_buffer)):
            r = self.n_step_buffer[i][2]
            reward += (self.gamma ** i) * r
            
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        
        
        self.n_step_buffer.popleft()
        
        return state, action, reward, next_state, done

    def train_step(self):
        
        if self.memory.tree.n_entries < BATCH_SIZE:
             return

        batch, idxs, weights = self.memory.sample(BATCH_SIZE)
        
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        curr_q = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_action = self.policy_net(next_state).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_state).gather(1, next_action)
            
            gamma_n = self.gamma ** self.n_step
            expected_q = reward + (gamma_n * next_q * (1 - done))

        loss_elementwise = (curr_q - expected_q).pow(2) 
        loss = (loss_elementwise * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        diff = torch.abs(curr_q - expected_q).detach().cpu().numpy()
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            error = diff[i][0]
            self.memory.update(idx, error)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())