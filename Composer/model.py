import torch.nn as nn
import torch
import math 
import librosa

# Constants
NUM_MIDI_NOTES = 128 
# Extra Tokens:
# SEP - Separator between note bundles
# EOS - End of Sequence
# MASK - Mask Token for prediction. Acts as padding as well.

VOCABULARY = {librosa.midi_to_note(i) : i  for i in range(0, 128)}
VOCABULARY["SEP"] = len(VOCABULARY)
VOCABULARY["EOS"] = len(VOCABULARY)
VOCABULARY["MASK"] = len(VOCABULARY)

VELOCITY_VALUES = 128

class Config:
    def __init__(self):
        self.context_len = 1024

        self.note_branch_layers = 4
        self.velocity_branch_layers = 4
        self.duration_branch_layers = 4
        self.time_branch_layers = 4

        self.note_embedding_dims = 16
        self.velocity_embedding_dims = 8

class DiscreteEmbedding(nn.Module):
    def __init__(self, context_len, embedding_dims, vocabulary_size, dropout):
        super(DiscreteEmbedding, self).__init__()
        self.positional = nn.Embedding(num_embeddings=context_len, embedding_dim=embedding_dims)
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dims, sparse=True)
        self.dropout = dropout
    
    def forward(self, x, _pos):
        embed = self.embedding(x)
        _pos = self.positional(_pos)
        embed = self.dropout(_pos + embed)

        return embed
    
class ContinuousEmbedding(nn.Module):
    def __init__(self, context_len, dropout):
        super(ContinuousEmbedding, self).__init__()
        self.dropout = dropout
    
    def forward(self, x, _pos):
        embed = self.dropout(x)

        return embed

class ComposeNet(nn.Module):
    def __init__(self, 
        config: Config = Config(),
        device = "cuda", 
    ):
        super(ComposeNet, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.note_embedding = DiscreteEmbedding(config.context_len, config.note_embedding_dims, len(VOCABULARY), dropout=self.dropout)
        self.velocity_embedding = DiscreteEmbedding(config.context_len, config.velocity_embedding_dims, VELOCITY_VALUES, dropout=self.dropout)

        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(config.context_len)
        self.note_branch = [
            nn.TransformerEncoderLayer(
                d_model=config.note_embedding_dims,
                nhead=8,
                dim_feedforward=16, 
                device = device
            ) 
            for _ in range(0, config.note_branch_layers)
        ]

        self.velocity_branch = [
            nn.TransformerEncoderLayer(
                d_model=config.velocity_embedding_dims, 
                nhead=4, 
                dim_feedforward=8, 
                device = device
            ) 
            for _ in range(0, config.velocity_branch_layers)
        ]

        self.note_linear = nn.Linear(16, len(VOCABULARY), device = device)
        self.velocity_linear = nn.Linear(8, VELOCITY_VALUES, device = device) 

        self.to(device=device)
        self._device = device
        self._context_len = config.context_len

    # Dimensions of x are (Batches, Note Sequences)
    # Note Sequences MUST have size < context_len 
    def forward(self, x):
        x_notes = x['notes']
        x_velocities = x['velocities']
        x_durations = x['durations']
        
        _, tn = x_notes.size()
        _, tv = x_velocities.size()
        _, td = x_durations.size()

        assert tn <= self._context_len, f"Cannot forward sequence of length {tn}, block size is only {self._context_len}"
        assert tn == tv, f"Length of velocities not equal to length of notes, {tn} != {tv}"
        assert tn == td, f"Length of durations not equal to length of notes, {tn} != {td}"
        
        # Pre-processing
        x_durations = torch.reshape(x_durations, (-1, td, 1))

        # Array for positionally embedded notes
        _pos = torch.arange(0, tn, dtype=torch.long, device=self._device)

        note_embed = self.note_embedding(x_notes, _pos)
        velocity_embed = self.velocity_embedding(x_velocities, _pos)
        duration_embed = x_durations # Temporary

        y_notes = note_embed  
        y_velocity = velocity_embed
        y_duration = duration_embed

        # Note branch
        for i, layer in enumerate(self.note_branch):
            y_notes = layer(y_notes, src_mask = self.causal_mask, is_causal = True)

        # Velocity Branch
        for i, layer in enumerate(self.velocity_branch):
            y_velocity = layer(y_velocity, src_mask = self.causal_mask, is_causal = True)

        # Duraton Branch

        # Get the next note
        y_notes = self.note_linear(y_notes[:, -1, :])
        y_notes = torch.softmax(y_notes, 1)

        # Get the next velocity
        y_velocity = self.velocity_linear(y_velocity[:, -1, :])
        y_velocity = torch.softmax(y_velocity, 1)

        # Get the next duration

        # Return the outputs
        # y_notes contains the logits of the next note
        # y_velocity contains the logits of the velocities
        # y_duration contains the durations
        return y_notes, y_velocity, y_duration
