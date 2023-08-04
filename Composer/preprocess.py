from models.data_split import VOCABULARY
import pretty_midi as pm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import math 

# Just to remove any warnings about pretty_midi. Please remove this when testing 
import warnings
warnings.filterwarnings("ignore")

# Bundle all note events together based on their time offset from each other
# Returns an array with dimensions (instruments, bundles per instrument)
def bundle_events(midi : pm.PrettyMIDI, instrument : pm.Instrument):
    bundles = []
    i = 0
    notes_list = sorted(instrument.notes, key=lambda x: (x.start, x.end))

    while i < len(notes_list):
        bundle = []  

        bnote :pm.Note = None   
        while i < len(notes_list): 
            note : pm.Note = notes_list[i]

            if len(bundle) == 0:
                bundle.append(note)
                bnote = note 
            elif math.isclose(midi.time_to_tick(note.start), midi.time_to_tick(bnote.start), rel_tol=1e-7) and \
                math.isclose(midi.time_to_tick(note.end), midi.time_to_tick(bnote.end), rel_tol=1e-7):
                bundle.append(note)
            else: 
                bundle = sorted(bundle, key = lambda x : x.pitch)
                bundles.append(bundle)
                break
            i = i + 1

    bundles = sorted(bundles, key=lambda x: x[0].start)
    return bundles

# Convert bundles to a list of notes, durations and velocities. 
# Each bundle note and velocity is separated from each other.
# Each duration corresponds to the duration of the note
# Each velocity is normalized to be between 0 and 1 (by dividing by 127)

def create_lists(instrument):    
    notes = []
    durations = []
    velocities = []
    times = []
    
    current_time = 0

    # Append beginning of sequence tok
    notes.append(VOCABULARY['BOS'])
    durations.append(0)
    velocities.append(0)
    times.append(0)

    for i, b in enumerate(instrument):
        for x in b: 
            notes.append(x.pitch)
            velocities.append(int(x.velocity))



        if not (math.isclose(b[0].start, current_time, rel_tol=1e-7)):
            notes.append(VOCABULARY["BEAT"])    # Add a special beat token whenever two groups have the same onset time
            times.append(b[0].start - current_time ) # Append the delta time otherwise.
        else: 
            # Add the end of the last chord
            notes.append(VOCABULARY["SEP"])
            durations.append(b[0].duration)  # Note: Every duration bounded by each SEP is assumed to be the same
        
        current_time = b[0].start

    notes.append(VOCABULARY["EOS"])
    durations.append(0)
    velocities.append(0)
    times.append(0)

    return np.array(notes), np.array(durations), np.array(velocities), np.array(times)

def normalize_to_beats(instrument_arr, ticks_per_beat):
    return np.round(instrument_arr / ticks_per_beat, 4)

def remove_bad_tracks(midi : pm.PrettyMIDI):
    new_midi = pm.PrettyMIDI(resolution = midi.resolution)

    for inst in midi.instruments:
        inst : pm.Instrument = inst

        # Ignore drums
        if inst.is_drum:
            continue 

        inst_class = pm.program_to_instrument_class(inst.program)

        # Ignore instruments that are not relevant
        if inst_class in ["Ethnic", "Percussive", "Sound Effects"]:
            continue 

        new_midi.instruments.append(inst)
    
    return new_midi

def flatten_notes(midi : pm.PrettyMIDI):
    notes_list = []

    for inst in midi.instruments:
        notes_list.extend(inst.notes)

    notes_list = sorted(notes_list, key=lambda x: x.start)

    flattened_midi = pm.PrettyMIDI(resolution = midi.resolution)
    instrument = pm.Instrument(0)

    instrument.notes = notes_list
    flattened_midi.instruments.append(instrument)

    return flattened_midi

def process_midi(path):   
    try: 
        midi_file = pm.PrettyMIDI(path)
    except: 
        return None 
    
    midi_file.remove_invalid_notes()

    # Flatten the midi file
    midi_file = remove_bad_tracks(midi_file)
    midi_file = flatten_notes(midi_file)

    instrument_notes = []
    instrument_durations = []
    instrument_velocities = []
    instrument_times = []
    instrument_names = []

    for _, instrument in enumerate(midi_file.instruments):
        instrument_bundles = bundle_events(midi_file, instrument)
        notes, durations, velocities, times = create_lists(instrument_bundles)

        durations = normalize_to_beats(durations, 1.0 / midi_file.resolution)
        times = normalize_to_beats(times, 1.0 / midi_file.resolution)
        
        instrument_notes.append(notes)
        instrument_durations.append(durations)
        instrument_velocities.append(velocities)
        instrument_times.append(times)
        instrument_names.append(instrument.program)

    return np.array(instrument_notes, dtype=object), \
        np.array(instrument_durations, dtype=object), \
        np.array(instrument_velocities, dtype=object), \
        np.array(instrument_times, dtype=object), \
        np.array(instrument_names, dtype=object)


def make_dataset(midis, file_name: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=["name", "instrument", "notes", "durations", "velocities", "times"])

    with tqdm(midis, unit="files") as tfiles:
        for i, mid in enumerate(tfiles):
            tfiles.set_description("Processing Files...")

            out = process_midi(mid,)

            tfiles.set_postfix_str(mid)

            if out is None: 
                tfiles.set_postfix_str(mid + "Skipping corrupted file...")
                continue 
            
            n, d, v, t, inst  = out

            for i in range(0, len(n)):
                if (len(n[i]) > 100): # Do not include entries with less than this many notes
                    df.loc[len(df.index)] = [mid.split('/')[-1], 
                                            inst[i],
                                            n[i].tolist(), 
                                            d[i].tolist(), 
                                            v[i].tolist(),
                                            t[i].tolist()]

    df.to_csv(file_name, index=False)
    return df