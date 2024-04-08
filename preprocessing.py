import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
import pretty_midi
import tempfile
from music21 import key, pitch, interval, converter, note


def midi_to_notes(midi_input) -> pd.DataFrame:
    pm = midi_input
    instrument = pm.instruments[0]

    # Prepare the notes data
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start) # Sorting the notes by start time
    
    if sorted_notes:
        prev_start = sorted_notes[0].start
    else:
        raise ValueError("No notes found in the MIDI data")

    # Extract note information
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['st_time'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    # Convert the notes data to a DataFrame
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str,velocity: int = 80, ) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['st_time'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def plot_piano_roll(midi_data, start_pitch, end_pitch, fs=100, time_span=10):
    # Generate a piano roll matrix
    piano_roll = midi_data.get_piano_roll(fs=fs)[start_pitch:end_pitch, :fs*time_span]  # Limiting to 30 seconds
    # Plot the piano roll
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size to your preference
    im = ax.imshow(piano_roll, aspect='auto', interpolation='nearest', cmap='viridis')  # Using a more colorful cmap
    #ax.set_yticks(np.arange(0, end_pitch - start_pitch, step=1))
    #ax.set_yticklabels(np.arange(start_pitch, end_pitch, step=1))
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Pitch")
    
    # Adding a colorbar
    plt.colorbar(im, ax=ax)

    # Making sure that the x-axis represents time in seconds
    x_ticks = np.arange(0, fs*time_span+1, fs)
    x_labels = [str(int(x/fs)) for x in x_ticks]  # converting frames to seconds
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Improve the visibility of the labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    return fig

def format_chord(chord_obj):
    root = chord_obj.root() # Get the root note's name

    # Determine the quality abbreviation, with a default of 'other'
    quality = chord_obj.quality
    chord_name = f"{root.name} {quality}"

    inversion_index = chord_obj.inversion()
    inversion_names = ["", "1st", "2nd", "3rd"]  # Shorthand for inversions, empty string for root position
    inversion_name = inversion_names[inversion_index] if inversion_index < len(inversion_names) else f"{inversion_index}th inversion"

    # Combine all parts without unnecessary hyphens
    #chord_name = f"{root} {quality}" if quality and quality != 'other' else root
    formatted_chord = f"{chord_name} {inversion_name}".strip()

    return formatted_chord



def transpose_to_input_key(generated_midi, input_midi):
    # Write the input PrettyMIDI object to a temporary MIDI file to analyze the key
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.mid') as temp_midi_file:
        input_midi.write(temp_midi_file.name)
        temp_midi_file.seek(0)
        # Parse the MIDI file with music21 for key analysis
        input_midi_stream = converter.parse(temp_midi_file.name)
        # Extract the key from the input MIDI
        input_key = input_midi_stream.analyze('key')
        print(f"Input Key: {input_key}")

    # Write the generated PrettyMIDI object to a temporary MIDI file to transpose
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.mid') as temp_midi_file:
        generated_midi.write(temp_midi_file.name)
        temp_midi_file.seek(0)
        # Parse the MIDI file with music21 for transposition
        generated_midi_stream = converter.parse(temp_midi_file.name)
        generated_key = generated_midi_stream.analyze('key')
        print(f"Generated Key: {generated_key}")

        # Decide whether to use the relative major or minor if necessary
        if input_key.mode != generated_key.mode:
            if input_key.mode == 'major':
                generated_key = key.Key(generated_key.tonic.name, 'major')
            elif input_key.mode == 'minor':
                generated_key = key.Key(generated_key.tonic.name, 'minor')

        # Calculate the interval for transposition
        transposition_interval = interval.Interval(generated_key.tonic, input_key.tonic)
        print(f"Transposing by interval: {transposition_interval}")

        # Calculate the interval for transposition in semitones
        input_pitch = pitch.Pitch(input_key.tonic.nameWithOctave)
        generated_pitch = pitch.Pitch(generated_key.tonic.nameWithOctave)
        transposition_interval_semitones = input_pitch.midi - generated_pitch.midi
        transposition_interval = interval.ChromaticInterval(transposition_interval_semitones)
        print(f"Transposing by semitones: {transposition_interval_semitones}")
        
        # Transpose the generated MIDI stream
        transposed_stream = generated_midi_stream.transpose(transposition_interval)
        # Write the transposed stream back to the temporary file
        transposed_stream.write('midi', temp_midi_file.name)
        # Read the transposed MIDI file with PrettyMIDI
        transposed_pretty_midi = pretty_midi.PrettyMIDI(temp_midi_file.name)

    return transposed_pretty_midi


def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature, seq_length, vocab_size) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, st_time, duration), using a trained sequence model."""
    assert temperature > 0

    # Select the last seq_length notes
    if notes.shape[0] > seq_length:
        input_notes = notes[-seq_length:]
    else:
        input_notes = notes

    # Normalize pitch values assuming pitch is in the first column
    input_notes = input_notes.copy()  # to avoid modifying the original array
    input_notes[:, 0] /= vocab_size

    # Add batch dimension
    inputs = np.expand_dims(input_notes, axis=0)

    # Get predictions from the model
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    st_time = predictions['st_time']
    duration = predictions['duration']

    # Apply temperature scaling to pitch logits and get the predicted pitch
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)

    # Constrain pitch to the range of piano keys
    piano_key_min = 21
    piano_key_max = 108
    pitch = tf.clip_by_value(pitch, piano_key_min, piano_key_max)

    # Ensure `st_time` and `duration` values are non-negative
    st_time = tf.squeeze(st_time, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    st_time = tf.maximum(0, st_time)
    duration = tf.maximum(0, duration)

    return int(pitch.numpy()), float(st_time.numpy()), float(duration.numpy())

def trim_midi(midi_data, max_length=30):
    """Trim the PrettyMIDI object to the first max_length seconds."""
    for instrument in midi_data.instruments:
        # Remove notes that start after the max_length or end after max_length
        instrument.notes = [note for note in instrument.notes if note.start < max_length and note.end <= max_length]
    return midi_data

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)
  
def load_model():
    model = tf.keras.models.load_model('hybrid_model.h5', custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})
    return model

