import numpy as np 
import pandas as pd
import os
import tempfile
from music21 import converter, chord
from preprocessing import midi_to_notes, format_chord, notes_to_midi,predict_next_note, load_model, transpose_to_input_key

vocab_size = 128
seq_length = 25
key_order = ['pitch', 'st_time', 'duration']
temperature = 2.0
num_predictions = 120

def extract_key_and_chords(pretty_midi_data):
    notes_at_ticks = {}
    for instrument in pretty_midi_data.instruments:
        for note_data in instrument.notes:
            tick = int(note_data.start * pretty_midi_data.resolution)
            if tick not in notes_at_ticks:
                notes_at_ticks[tick] = set()
            notes_at_ticks[tick].add(note_data.pitch)

    chords = [notes for notes in notes_at_ticks.values() if len(notes) > 1]
    chord_objects = [chord.Chord(notes) for notes in chords]

    # Format the chords into a readable format
    formatted_chords = [format_chord(chord_obj) for chord_obj in chord_objects]

    # Only take the first 20 chords
    formatted_chords = formatted_chords[:20]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    temp_midi_path = temp_file.name
    temp_file.close()
    
    try:
        pretty_midi_data.write(temp_midi_path)
        midi_stream = converter.parse(temp_midi_path)
        midi_key = midi_stream.analyze('key')
    finally:
        os.remove(temp_midi_path)

    return formatted_chords, midi_key

def generate_notes_sequence(input_notes, num_predictions=120):
    """
    Generate a sequence of notes based on the input notes array.

    :param input_notes: Array of input notes (typically the start of a sequence)
    :param num_predictions: Number of notes to generate
    :return: DataFrame containing generated notes and their timings
    """
    # Load your trained model
    model = load_model()
    
    # Normalize the input notes similar to the model's training
    input_notes = np.stack([input_notes[key] for key in key_order], axis=1)
    input_notes = input_notes[:seq_length] / np.array([vocab_size, 1, 1])

    generated_notes = []
    prev_start = 0

    for _ in range(num_predictions):
        pitch, st_time, duration = predict_next_note(input_notes, model, temperature, seq_length, vocab_size)
        start = prev_start + st_time
        end = start + duration
        input_note = (pitch, st_time, duration)

        generated_notes.append((*input_note, start, end))

        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))
    return generated_notes_df

# Placeholder function to represent melody generation
def generate_melody(midi_data):
    # Analyze the key of the input MIDI
    input_notes = midi_to_notes(midi_data)
    generated_notes = generate_notes_sequence(input_notes)
    generated_midi_data = notes_to_midi(generated_notes, "generated.mid", "Acoustic Grand Piano")
    generated_melody = transpose_to_input_key(generated_midi_data, midi_data)
    return generated_melody