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
VOCABULARY["BEAT"] = len(VOCABULARY)
VOCABULARY["EOS"] = len(VOCABULARY)
VOCABULARY["PAD"] = len(VOCABULARY)


VELOCITY_VALUES = 128

class MidiDataset(Dataset):
    def __init__(self, df : pd.DataFrame, context_len, train_samples = 1e5, validate_samples = 1e4, start_length = -1):
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

        self._samples = train_samples
        self._train_samples = train_samples
        self._validate_samples = validate_samples

        self._data_points = len(self.notes)

        if start_length == -1:
            self._start_length = context_len
        else:
            self._start_length = start_length

    def __len__(self):
        return self._samples

    def __getitem__(self, idx):
        index = np.random.randint(0, self._data_points)
        note_slice = self.notes[index]
        
        # Data Augmentation technique: Transpose the notes
        transpose_semitones = np.random.randint(-12, 13)  
        note_slice = self.transpose(note_slice, transpose_semitones)

        # max_length = min(self._start_length, len(note_slice) - 1)
        # min_length = self._start_length
        length = self._start_length
        offset = np.random.randint(0, len(note_slice) - length)
        gt_note = note_slice[offset + length]
        
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
                    constant_values=VOCABULARY['PAD']
                    )


        # A sample taken from this slice
        sample = {
            "notes": notes
        }

        # The expected outputs
        gt = {
            "notes": np.array(gt_note, dtype=np.int64)
        }
        return sample, attn_idx, gt
    
    # Transposes the notes by shifting up a semitone. This makes the model more invariant to key signatures.
    def transpose(self, notes, semitones):
        A = np.array(notes)
        L = A + semitones >= 0
        H = A + semitones < NUM_MIDI_NOTES 

        M = A < NUM_MIDI_NOTES
        M = A * M >= 0 

        return (A + M * H * semitones) * L

    def make_note_logit(self, note):
        logit = np.zeros(len(VOCABULARY), dtype = float)
        logit[note] = 1.0
        return logit.astype(float)
    
    def set_training(self):
        self._samples = self._train_samples

    def set_validation(self):
        self._samples = self._validate_samples