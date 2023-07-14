from torch.utils.data import Dataset

class MidiDataset(Dataset):
    def __init__(self, df, context_len):
        self.notes = df['notes']
        self.durations = df['durations'] 
        self.times = df['times']
        self.velocities = df['velocities']

        self.context_len = context_len
    
    def __len__(self):
        return len(self.notes)
    
    def __getitem__(self, idx):
        notes = self.notes[idx][: self.context_len]
        durations = self.durations[idx][: self.context_len]
        times = self.times[idx][: self.context_len]
        velocities = self.velocities[idx][: self.context_len]

        sample = {
            "notes": notes, 
            "durations": durations, 
            "times": times, 
            "velocities": velocities 
        }
        return sample