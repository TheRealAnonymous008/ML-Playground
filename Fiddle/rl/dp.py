import gymnasium as gym
import numpy.random as random
import numpy as np 
import itertools

class Policy: 
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.policy = np.zeros(state_count, dtype=np.int8)
        self.states = []

        if type(self.state_count) is tuple:
            axes = [[i for i in range(0, x)] for x in self.state_count]
            self.states =itertools.product(*axes)
                
        else:
            self.states = [i for i in range(0, self.state_count)] 

    def sample(self, state):        
        return self.policy[state]

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
            self.policy.policy[s] = np.argmax(self.Q(s, env))

        return self.policy

    def Q(self, s, env : gym.Env):
        action_values = np.zeros(self.policy.action_count)

        for a in range(0, self.policy.action_count):
            try: 
                p = env.P[s][a]
            except:
                p = 1

            for probability, next_state, reward, info in env.P[s][a]:
                action_values[a] += probability * (reward + self.discount_rate * self.V[next_state])
            
        return action_values
        