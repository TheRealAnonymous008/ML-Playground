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

class ComposeNet(nn.Module):
    def __init__(self, device = "cuda", context_len = 256):
        super(ComposeNet, self).__init__()
        self.positional = nn.Embedding(num_embeddings=context_len, embedding_dim=16)
        self.note_embedding = nn.Embedding(num_embeddings=len(VOCABULARY), embedding_dim=16, sparse=True)
        self.dropout = nn.Dropout(0.1)

        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(16)

        self.encoder_layers = [ \
            nn.TransformerEncoderLayer(d_model=16, nhead=8, dim_feedforward=16, device = device) 
            for _ in range(0, 2)
        ]
        
        self.linear = nn.Linear(16, len(VOCABULARY), device = device)

        self.to(device=device)
        self._device = device
        self._context_len = context_len

    # Dimensions of x are (Batches, Note Sequences)
    # Note Sequences MUST have size < context_len 
    def forward(self, x):
        _, t = x.size()

        # Array for positional embedding
        _pos = torch.arange(0, t, dtype=torch.long, device=self._device)
        embed = self.note_embedding(x)
        pos = self.positional(_pos)

        # Do embedding
        x = self.dropout(pos + embed)
        
        y = x
        
        for layer in self.encoder_layers:
            y = layer(y, src_mask = self.causal_mask, is_causal = True)
        
        y = self.linear(y[:, -1, :])
        y = torch.softmax(y, 1)
        return y
