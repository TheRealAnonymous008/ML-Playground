from torch.utils.data import Dataset
import numpy as np
import librosa


# Constants
NUM_MIDI_NOTES = 128 
# Extra Tokens:
# SEP - Separator between note bundles
# EOS - End of Sequence
# MASK - Mask Token for prediction. Acts as padding as well.

VOCABULARY = { librosa.midi_to_note(i) : i  for i in range(0, NUM_MIDI_NOTES)}
VOCABULARY["SEP"] = len(VOCABULARY)
VOCABULARY["REST"] = len(VOCABULARY)
VOCABULARY["EOS"] = len(VOCABULARY)
VOCABULARY["MASK"] = len(VOCABULARY)


VELOCITY_VALUES = 128

class MidiDataset(Dataset):
    def __init__(self, df, context_len, samples_per_track = 16):
        df['notes'] = df.notes.apply(lambda x: [int(y) for y in str(x).removeprefix('[').removesuffix(']').split(' ') if y.isnumeric()])
        df['velocities'] = df.velocities.apply(lambda x: [int(y) for y in str(x).removeprefix('[').removesuffix(']').split(' ') if y.isnumeric()])
        df['durations'] = df.durations.apply(lambda x: [float(y) for y in str(x).removeprefix('[').removesuffix(']').split(' ') if y.replace('.', '').replace('e+', '').replace('e-','').isnumeric()])

        self.notes = df['notes']
        self.durations = df['durations']
        self.velocities = df['velocities']

        self.context_len = context_len

        self._samples_per_track = samples_per_track
        self._data_points = len(self.notes)

    def __len__(self):
        return self._data_points * self._samples_per_track

    def __getitem__(self, idx):
        note_slice = self.notes[idx // self._samples_per_track]

        # Sample based on an offset.
        offset = np.random.randint(0, len(note_slice))
        slice_length = np.random.randint(0, self.context_len)

        notes = note_slice[offset: offset + slice_length]

        # Store counts
        count = len(notes) 

        # Perform padding
        # Note: Padding is necessary for batching. We specify an appropriate attention mask
        pad_toks = self.context_len - count
        attn_mask = np.concatenate([np.array([float("-inf") for _ in range(0, count)]), 
                                    np.zeros(pad_toks, np.float32)], dtype=np.float32)

        if pad_toks > 0:
            notes = np.pad(notes, 
                    [(0, pad_toks)],
                    mode='constant',
                    constant_values=VOCABULARY['EOS']
                    )

        # A sample taken from this slice
        sample = {
            "notes": notes.astype(np.int32)
        }

        gt_note = 0 
        if pad_toks > 0: 
            gt_note = VOCABULARY["EOS"]
        else: 
            gt_note = note_slice[count + 1]

        # The expected outputs
        gt = {
            "notes": self.make_note_logit(gt_note)
        }
        return sample, attn_mask, gt
    

    def make_note_logit(self, note):
        logit = np.zeros(len(VOCABULARY))
        logit[note] = 1
        return logit
        