import streamlit as st
import pretty_midi
from generate import extract_key_and_chords, generate_melody
from preprocessing import plot_piano_roll

# Style css

# Set page title and favicon
st.set_page_config(page_title="The Chorder", page_icon="🎵")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>" , unsafe_allow_html=True)

local_css("style/style.css")

def main():
    st.title("🎹 The Chorder - MIDI Melody Generator and Chord Predictor")

    # Provide a sample MIDI file for users to experiment with
    sample_midi = st.sidebar.radio("Use a sample MIDI or upload your own:", ("Upload", "Sample"))
    if sample_midi != "Upload":
        if 'sample_midi_path' not in st.session_state or st.session_state.sample_midi_path != sample_midi:
            st.session_state.sample_midi_path = sample_midi
            sample_midi_path = "sampleMIDI.midi" 
            st.session_state.input_midi = pretty_midi.PrettyMIDI(sample_midi_path)
            st.sidebar.success("Loaded Sample MIDI successfully!")
            if 'generated_midi_data' in st.session_state:
                del st.session_state['generated_midi_data']
    else:
        midi_file = st.file_uploader("Upload MIDI file", type=['mid', 'midi'], key="midi_upload")
        if midi_file is not None:
            if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != midi_file.name:
                st.session_state.uploaded_file = midi_file.name
                st.session_state.input_midi = pretty_midi.PrettyMIDI(midi_file)
                st.sidebar.success(f"Loaded `{midi_file.name}` successfully!")
                if 'generated_midi_data' in st.session_state:
                    del st.session_state['generated_midi_data']

    input_midi = st.session_state.get('input_midi', None)

    if input_midi:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input MIDI Piano Roll")
            fig_input = plot_piano_roll(input_midi, 21, 108)
            st.pyplot(fig_input)
            
            # Display Roman numeral chords and keys
            st.markdown("Input Chords and Keys:", unsafe_allow_html=True)
            input_chords, input_key = extract_key_and_chords(input_midi)
            
            # Display the generated key
            st.write("Input Key:", input_key.tonic.name, input_key.mode)

            # Display the generated chords as separate lines of text
            st.write("Input Chords:")
            for chord in input_chords:
                st.text(chord)
        
        with col2:
            st.subheader("Output MIDI Piano Roll")
            generated_midi_data = st.session_state.get('generated_midi_data', None)
            if generated_midi_data:
                fig_output = plot_piano_roll(generated_midi_data, 21, 108)
                st.pyplot(fig_output)
                
                # Display Roman numeral chords and keys
                st.write("Generated Chords and Keys:")
                generated_chords, generated_key = extract_key_and_chords(generated_midi_data)
                
                # Display the generated key
                st.write("Generated Key:", generated_key.tonic.name, generated_key.mode)
                
                # Display the generated chords as separate lines of text
                st.write("Generated Chords:")
                for chord in generated_chords:
                    st.text(chord)
                

    else:
        st.error("Please upload an input MIDI file or choose a sample to proceed.")

    if st.button("Generate Melody"):
        if input_midi:
            generated_midi_data = generate_melody(input_midi)
            st.session_state['generated_midi_data'] = generated_midi_data
            st.rerun()

if __name__ == "__main__":
    main()

