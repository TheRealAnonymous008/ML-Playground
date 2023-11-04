import numpy as np
import itertools

class Policy: 
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.states = []
        self.actions = []
        if type(self.state_count) is not int:
            axes = [[i for i in range(0, x)] for x in self.state_count]
            self.states =[it for it in itertools.product(*axes)]

            policy_shape = self.state_count
            policy_shape.append(self.action_count)

            self.policy = np.zeros(policy_shape, dtype = np.float16)
                
        else:
            self.states = [i for i in range(0, self.state_count)] 
            self.policy = np.zeros((self.state_count, self.action_count), dtype=np.float16)

        self.actions = [i for i in range(0, self.action_count)]

    def sample(self, state):        
        return np.random.choice(self.actions, p = self.policy[state])
    
    def get_total_states(self):
        return len(self.states)
    
    def get_total_actions(self):
        return len(self.actions)