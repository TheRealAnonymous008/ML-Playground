# Id is either 1 or -1 for X and O respetively
from game import Game, GameState
from collections import namedtuple, deque
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 32
GAMMA = 0.5
def serialize_board(board):
    serial = []

    for row in board:
        for x in row:
            serial.append(x)

    return serial

class DeepQ(nn.Module):
    def __init__(self):
        super(DeepQ, self).__init__()

        # Inputs are the 81 inputs in the serialized board 
        # Plus the current box 
        # Plus the current player
        self.l1 = nn.Linear(83, 64, True, dtype=torch.float16),
        self.l2 = nn.Linear(64, 16, True, dtype=torch.float16),
        self.l3 = nn.Linear(16, 1, True, dtype=torch.float16),


    def forward(self, state : GameState):
        inputs = []
        inputs = serialize_board(state.board)
        inputs.append(state.current_box)
        inputs.append(state.current_player)

        input_tensor = torch.tensor(inputs, dtype=torch.float16, device="cuda")
        output = F.gelu(self.l1(input_tensor))
        output = F.gelu(self.l2(output))
        output = F.gelu(self.l3(output))

        return output

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Player:
    def __init__(self, id):
        self.id = id 
        self.initialize_brain()

    def initialize_brain(self):
        self.policy = DeepQ()
        self.policy.to("cuda")

        self.target = DeepQ()
        self.target.to("cuda")

        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.001, momentum=0.9)
        self.memory = ReplayMemory(10000)
    
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return 
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device="cuda", dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device="cuda")
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

    def play(self, game : Game):
        if game.get_current_player() != self.id: 
            print("Not my turn yet")
            return 
        
        else: 
            # Generate all possible moves
            successors = game.state.get_successors()
            # Evaluate moves
            best_succ = None 
            best_score = -100000

            for succ in successors: 
                r = self.eval(succ)
                if best_score < r: 
                    best_succ = succ 
                    best_score = r 

            return best_succ, best_score

    def eval(self, state: GameState):
        return self.policy(state).cpu().detach().numpy()[0]