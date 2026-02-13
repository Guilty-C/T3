import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        if hidden_dim > 0:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
             # Linear policy (Softmax regression) for simple bandits
            self.actor = nn.Sequential(
                nn.Linear(state_dim, action_dim),
                nn.Softmax(dim=-1)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 1)
            )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy

class LagrangianPPO:
    def __init__(self, state_dim, action_dim, constraint_threshold=0.1, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, entropy_coef=0.01, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_threshold = constraint_threshold
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy_old.act(state)
        return action, log_prob.item()
    
    def update(self, memory):
        # Convert memory to list of tensors
        rewards = []
        discounted_reward = 0
        for reward in reversed([m['reward'] for m in memory]):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) # Normalize
        
        old_states = torch.tensor(np.array([m['state'] for m in memory]), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array([m['action'] for m in memory]), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array([m['log_prob'] for m in memory]), dtype=torch.float32).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Loss = -min(surr1, surr2) + 0.5*MSE(state_value, reward) - 0.01*entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
