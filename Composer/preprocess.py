from dataset import VOCABULARY
import mido
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Bundle all note events together based on their time offset from each other
# Returns an array with dimensions (tracks, bundles per track)
def bundle_events(track : mido.MidiTrack):
    bundle = []
    bundles = []
    for msg in track:
        attrs = msg.dict()
        type = attrs["type"]
            
        if type == 'note_on' or type == 'note_off':
            if attrs['time'] == 0 or len(bundle) == 0:
                bundle.append(attrs)
            else:
                # Ensure that the first entry in the list has the time stamp info
                delta = bundle[0]['time']
                # Sort each entry in each bundle based on the note (in ascending order) 
                sorted_bundle = sorted(bundle, key=lambda x: int(x['note']))

                # Set all bundle entries to have time = 0 except for the first entry
                for b in sorted_bundle:
                    b['time'] = 0
                sorted_bundle[0]['time'] = delta 

                bundles.append(sorted_bundle)

                bundle.clear()
                bundle.append(attrs)

    return bundles

# Augment track bundles by removing note off / velocity = 0 events. 
def augment_track_bundles(track, verbose):
    aug = [[] for i in range(0, len(track))]
    time = 0

    # entries are of the form
    # note_id : {event, time_started, idx }
    state = {}

    # INitialize state 
    for i in range(0, 128): 
        state[i] = None

    for i, bundle in enumerate(track): 
        time += bundle[0]['time']
        for event in bundle: 
            if event['type'] == 'note_off' or event['velocity'] == 0:
                info = state[event['note']]

                if info == None: 
                    if verbose:
                        print("Poorly formatted key. Skipping")
                    continue 
                
                info['event']['duration'] = time - info['time']
                info['event']['time'] = info['time']

                if 'type' in info['event'].keys():
                    del info['event']['type']

                aug[info['idx']].append(info['event']) 
            else: 
                state[event['note']] = {'event': event, 'time': time, 'idx': i}

    # Remove any empty entries
    augmented = filter(lambda x : len(x) != 0, aug)

    return augmented

# Convert bundles to a list of notes, durations and velocities. 
# Each bundle note and velocity is separated from each other.
# Each duration corresponds to the duration of the note
# Each velocity is normalized to be between 0 and 1 (by dividing by 127)

def create_lists(track):    
    notes = []
    durations = []
    velocities = []
    
    current_time = 0
    last_note_time = 0

    for i, b in enumerate(track):
        current_time = b[0]['time']

        # Add a rest node 
        if i > 0: 
            notes.append(VOCABULARY['REST'])
            durations.append(current_time - last_note_time)
            velocities.append(0)

        longest_duration = -1
        for x in b: 
            notes.append(x['note'])
            durations.append(x['duration'])
            velocities.append(int(x['velocity']))

            longest_duration = max(longest_duration, x['duration'])

        last_note_time = current_time + longest_duration

        # Add the end of the last chord
        notes.append(VOCABULARY["SEP"])
        durations.append(0)
        velocities.append(0)

    notes.append(VOCABULARY["EOS"])
    durations.append(0)
    velocities.append(0)

    return np.array(notes), np.array(durations), np.array(velocities)

def normalize_to_beats(track_arr, ticks_per_beat):
    return track_arr / ticks_per_beat

def process_midi(path, verbose = False):
    if verbose:
        print("\"", path, "\"")
        
    try: 
        midi_file = mido.MidiFile(path)
        print("Tracks: ", len(midi_file.tracks))
    except: 
        print("Skipping corrupted file...")
        return None 

    track_notes = []
    track_durations = []
    track_velocities = []

    for _, track in enumerate(midi_file.tracks):
        track_bundles = bundle_events(track)
        augmented_bundles = augment_track_bundles(track_bundles, verbose=verbose)
        notes, durations, velocities = create_lists(augmented_bundles)

        durations = normalize_to_beats(durations, midi_file.ticks_per_beat)
        
        track_notes.append(notes)
        track_durations.append(durations)
        track_velocities.append(velocities)

    return np.array(track_notes, dtype=object), \
        np.array(track_durations, dtype=object), \
        np.array(track_velocities, dtype=object)

def make_dataset(midis, file_name: str, verbose = False) -> pd.DataFrame:

    df = pd.DataFrame(columns=["name", "notes", "durations", "velocities"])

    for mid in midis:
        out = process_midi(mid, verbose)
        if out is None: 
            continue 
        
        n, d, v  = out

        for i in range(0, len(n)):
            if (len(n[i]) > 1): # Do not include entries with no notes.
                df.loc[len(df.index)] = [mid.split('/')[-1], n[i], d[i], v[i]]

    df.to_csv(file_name, index=False)
    return df