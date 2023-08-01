from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa


# Constants
NUM_MIDI_NOTES = 128 
# Extra Tokens:
# SEP - Separator between note bundles
# EOS - End of Sequence
# MASK - Mask Token for prediction. Acts as padding as well.

VOCABULARY = { librosa.midi_to_note(i) : i  for i in range(0, NUM_MIDI_NOTES)}
VOCABULARY["BOS"] = len(VOCABULARY)
VOCABULARY["SEP"] = len(VOCABULARY)
VOCABULARY["EOS"] = len(VOCABULARY)
VOCABULARY["MASK"] = len(VOCABULARY)


VELOCITY_VALUES = 128

class MidiDataset(Dataset):
    def __init__(self, df : pd.DataFrame, context_len, train_samples = 16, validate_samples = 4, start_length = -1):
        df['notes'] = df.notes.apply(lambda x: [int(y) for y in str(x).removeprefix('[').removesuffix(']').split(', ') if y.isnumeric()])
        df['velocities'] = df.velocities.apply(lambda x: [int(y) for y in str(x).removeprefix('[').removesuffix(']').split(', ') if y.isnumeric()])
        df['durations'] = df.durations.apply(lambda x: [float(y) for y in str(x).removeprefix('[').removesuffix(']').split(', ') if y.replace('.', '').replace('e+', '').replace('e-','').isnumeric()])
        df['times'] = df.times.apply(lambda x: [float(y) for y in str(x).removeprefix('[').removesuffix(']').split(', ') if y.replace('.', '').replace('e+', '').replace('e-','').isnumeric()])

        # Instrument info
        self.instruments = df['instrument']

        self.notes = df['notes']
        self.durations = df['durations']
        self.velocities = df['velocities']
        self.times = df['times']

        self.context_len = context_len

        self._samples_per_track = train_samples
        self._train_samples = train_samples
        self._validate_samples = validate_samples

        self._data_points = len(self.notes)

        if start_length == -1:
            self._start_length = context_len
        else:
            self._start_length = start_length

    def __len__(self):
        return self._data_points * self._samples_per_track

    def __getitem__(self, idx):
        note_slice = self.notes[idx // self._samples_per_track]

        length = np.random.randint(1, self._start_length)

        # Sample based on an offset.
        if len(note_slice) <= length:  # Case where ctx window is smaller than slice
            offset = 0 
        else:  # Case where ctx window is >= slice
            offset = np.random.randint(0, len(note_slice) - length)

        if offset + length < len(note_slice):
            gt_note = note_slice[offset + length]
        else:
            gt_note = VOCABULARY['EOS']
        
        notes = note_slice[offset: offset + length]
        notes = notes[:length]

        # Store counts
        count = len(notes) 

        # Perform padding
        # Note: Padding is necessary for batching. We specify an appropriate attention mask
        pad_toks = self.context_len - count
        attn_idx = count

        if pad_toks > 0:
            notes = np.pad(notes, 
                    [(0, pad_toks)],
                    mode='constant',
                    constant_values=VOCABULARY['EOS']
                    )

        # A sample taken from this slice
        sample = {
            "notes": notes
        }

        # The expected outputs
        gt = {
            "notes": self.make_note_logit(gt_note)
        }
        return sample, attn_idx, gt
    

    def make_note_logit(self, note):
        logit = np.zeros(len(VOCABULARY), dtype = np.float16)
        logit[note] = 1.0
        return logit
    
    def set_training(self):
        self._samples_per_track = self._train_samples

    def set_validation(self):
        self._samples_per_track = self._validate_samples