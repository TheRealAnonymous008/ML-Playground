import torch.nn as nn
import numpy as np
import torch

from .data_split import MidiDataset, VOCABULARY, NUM_MIDI_NOTES


class Config:
    def __init__(self):
        self.context_len = 256

        self.note_branch_layers = 16
        self.velocity_branch_layers = 4
        self.duration_branch_layers = 4
        self.time_branch_layers = 4

        self.note_embedding_dims = 16 # Potentially change this to 48
        self.velocity_embedding_dims = 8

class DiscreteEmbedding(nn.Module):
    def __init__(self, context_len, embedding_dims, vocabulary_size, dropout):
        super(DiscreteEmbedding, self).__init__()
        self.positional = nn.Embedding(num_embeddings=context_len, embedding_dim=embedding_dims)
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dims, sparse=False)
        self.dropout = dropout
    
    def forward(self, x, _pos):
        embed = self.embedding(x)
        _pos = self.positional(_pos)
        embed = self.dropout(_pos + embed)

        return embed

class NoteComposeNet(nn.Module):
    def __init__(self, 
        config: Config = Config(),
        device = "cuda", 
    ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.note_embedding = DiscreteEmbedding(config.context_len, config.note_embedding_dims, len(VOCABULARY), dropout=self.dropout)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(config.context_len, device=device)
        self.register_buffer("causal_mask", causal_mask)
        
        self.note_branch = torch.nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=config.note_embedding_dims,
                nhead=8,
                dim_feedforward=config.note_embedding_dims, 
                activation= torch.nn.functional.gelu,
                device = device
            ) 
            for _ in range(0, config.note_branch_layers)]
        )

        self.note_linear = nn.Linear(config.note_embedding_dims, len(VOCABULARY), device = device)

        self.to(device=device)
        self._device = device
        self._context_len = config.context_len


    # Dimensions of x are (Batches, Note Sequences)
    # Note Sequences MUST have size < context_len 
    def forward(self, x_notes):
        _, tn = x_notes.size()
        assert tn <= self._context_len, f"Cannot forward sequence of length {tn}, block size is only {self._context_len}"
        # Array for positionally embedded notes
        _pos = torch.arange(0, tn, dtype=torch.long, device=self._device)

        note_embed = self.note_embedding(x_notes, _pos)
        y_notes = note_embed 

        # Note branch
        for i, layer in enumerate(self.note_branch):
            y_notes = layer(y_notes, src_mask = self._buffers['causal_mask'], is_causal = True)

        # Get the next note
        y_notes = self.note_linear(y_notes[:, -1, :])

        # Return the next note
        return y_notes

    # Returns a number corresponding to the note generated.
    # Inputs are in array form
    @torch.no_grad()
    def generate(self, inputs, max_len = 10, temperature = 1.0, top_p = -1, prior_notes = None, prior_weight = 1.0):
        assert top_p <= 1.0

        input_toks = inputs[:self._context_len]
        outputs = []

        last_tok = inputs[-1]

        ALL_TOKS = [i for i in range(0, len(VOCABULARY))]

        if prior_notes is not None: 
            prior_logits_tensor = torch.tensor(prior_notes * prior_weight, device=self._device)

        for i in range(0, max_len):
            toks = torch.tensor([input_toks], device=self._device)

            output_logits = self.forward(toks)

            # Apply priors 
            if prior_notes is not None: 
                output_logits += prior_logits_tensor

            output_logits = torch.softmax(output_logits / temperature, 1)
            output_logits = output_logits.cpu().detach().numpy()
            
            # Cheat a bit and sample only from the logits above the current token (assuming it's a note)
            
            output_logits = output_logits[0]
            for i, logit in enumerate(output_logits):
                # Follow the format of the tokens
                if i < last_tok and last_tok < NUM_MIDI_NOTES:
                    output_logits[i] = 0
            
            output_logits[last_tok] = 0
                
            # Normalize
            sum_logits = sum(output_logits)
            output_logits /= sum_logits

            # Perform top_p sampling if top_p != -1
            if top_p > 0: 
                indexes = sorted(ALL_TOKS, key=lambda x: output_logits[x])
                low_p = 1.0 - top_p 

                cum_p = 0.0
                itr = 0

                while True: 
                    cum_p += output_logits[indexes[itr]]
                    if cum_p >= low_p:
                        break
                    output_logits[indexes[itr]] = 0 
                    itr += 1
                
                
            output_tok = torch.multinomial(torch.tensor(output_logits), num_samples=1)
            outputs.append(output_tok.item())
            last_tok = output_tok.item()

            input_toks = np.append(input_toks, output_tok)
            input_toks = input_toks[-self._context_len:]
        return outputs
    
    def detokenize(self, inputs):
        detoks = []
        keys = list(VOCABULARY.keys())
        for x in inputs:
            detoks.append(keys[int(x)])

        return detoks