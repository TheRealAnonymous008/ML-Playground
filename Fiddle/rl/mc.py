import gymnasium as gym
from .policy import Policy
import numpy as np

class MonteCarlo:
    def __init__(self, policy : Policy, env : gym.Env, sampling_policy : Policy | None = None):
        self.policy = policy
        if sampling_policy is None:
            self.sampling_policy = policy 
        else: 
            self.sampling_policy = sampling_policy

        self.Q = np.zeros(self.policy.policy.shape)
        self.returns = np.zeros(self.policy.policy.shape)
        self.counts = np.zeros(self.policy.policy.shape)

        self.env = env

        self.initialize_policy()

    def forward(self, iters = 100, discount = 0.95, epsilon = 0.5):
        for _ in range(iters):
            trajectory, sa_pairs = self.generate_episode()
            G = 0

            for i, sar in enumerate(reversed(trajectory)):
                S, A, R = sar
                G = discount * G + R 
                
                if (S, A) not in sa_pairs[:i]: 
                    self.returns[S][A] += G
                    self.counts[S][A] += 1

                    self.Q[S][A] = self.returns[S][A] / self.counts[S][A]
                    
                    maximums = np.argwhere(self.Q[S] == np.amax(self.Q[S])).flatten()
                    A = np.random.choice(maximums)

                    for a in self.policy.actions:
                        if a == A:
                            update = 1 - epsilon + epsilon / self.policy.action_count
                        else: 
                            update = epsilon / self.policy.action_count
                        self.policy.policy[S][a] = update
                        
    def generate_episode(self):
        trajectory = []
        state_action_pairs = []
        state, _ = self.env.reset()

        while True: 
            action = self.sampling_policy.sample(state)
            state, reward, isTerminated, _, _ = self.env.step(action)
            
            trajectory.append([state, action, reward])
            state_action_pairs.append((state, action))

            if isTerminated:
                break 

        return trajectory, state_action_pairs
    
    def initialize_policy(self):
        actions = 1.0 / self.policy.action_count
        for x in self.policy.states: 
            for a in self.policy.actions:
                self.policy.policy[x][a] = actions