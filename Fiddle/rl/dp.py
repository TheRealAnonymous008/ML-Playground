import gymnasium as gym
from .policy import Policy
import numpy as np

class ValueIteration: 
    def __init__(self, policy : Policy, discount_rate = 0.95):
        self.policy = policy
        self.V = np.zeros((policy.state_count))
        self.discount_rate = discount_rate

    # Value Iteeration assumes a known environment so we
    # have to pass the environment. 
    def forward(self, env : gym.Env, iters : int = 100):
        for _ in range(0, iters):
            for s in self.policy.states:
                self.V[s] = np.max(self.Q(s, env))

        for s in self.policy.states:
            optimal = np.argmax(self.Q(s, env))

            for a in range(self.policy.action_count):
                if a == optimal:
                    self.policy.policy[s][a] = 1.0
                else:
                    self.policy.policy[s][a] = 0.0

        return self.policy

    def Q(self, s, env : gym.Env):
        action_values = np.zeros(self.policy.action_count)

        for a in range(0, self.policy.action_count):
            for probability, next_state, reward, info in env.P[s][a]:
                action_values[a] += probability * (reward + self.discount_rate * self.V[next_state])
            
        return action_values
        