import streamlit as st

def documentation_page():
    """
    Documentation for the Music Generation Web Application
    """
    st.title('The Chorder Documentation')

    st.header('About')
    st.write('''
    This web app was developed as part of my final year research project to explore the capabilities of machine learning in music generation. It aims to provide an accessible platform for users to experience the intersection of technology and creativity in music. The features of this application include,
    ''')
    st.write('''
    - **MIDI File Upload:** Users can upload MIDI files as input for the music generation model.
    - **Music Generation:** The application uses a deep learning model to generate music based on the input MIDI file.
    - **Chord Progression Prediction:** Users can get chord progressions from the generated music.
    ''')

    st.subheader('Instructions')
    st.write('''
    - **Step 1:** Navigate to the Upload page to input your MIDI file.
    - **Step 2:** After uploading the MIDI file, time would be taken to generate chords from the input data.
    - **Step 2:** After the input chord generation, press the "Generate Melody" to initiate the music generation process.
    - **Step 3:** Once the generation process is complete, see the the generated chord progressions.
    ''')

    st.header('About')
    st.write('''
    Classical music generation presents a significant challenge due to its complexity and nuanced structures, making it difficult for artificial intelligence to produce compositions that truly resemble human-created music. In this context, chord progression prediction becomes a valuable tool, offering composers and producers musical direction and inspiration by suggesting harmonic sequences.

    To address this challenge, "The Chorder" utilizes a sophisticated deep learning methodology. It combines a multi-output hybrid architecture that integrates Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks. This system is trained on segments of MIDI files, where the input consists of the individual notes in each segment, and the output captures the essential attributes of these musical notes. By analyzing the nuanced patterns in these excerpts, "The Chorder" can generate music that not only mimics the classical style but also provides insights into potential chord progressions. This is achieved through a detailed extraction and analysis of chord data from the AI-generated music, thus aiding in the creative process by offering harmonically coherent suggestions.
    ''')
    
    st.subheader('System Architecture')
    st.write('''
    - Data preprocessing extracted three key features: pitch, start time, and duration from MIDI files.
    - The model, built with 1 CNN and 3 LSTM layers, captured both immediate patterns and long-term dependencies in music data.
    - Optimization was performed using Adam optimizer, with training over 20 epochs and early stopping to prevent overfitting.
    - Hyperparameter tuning was executed using Keras-Tuner’s Hyperband, optimizing layer counts, filters, dropout rates, learning rate, and regularization.
    - The final model, refined with optimal hyperparameters, was retrained and integrated into the Streamlit application for music generation and chord progression prediction.
    ''')


if __name__ == '__main__':
    documentation_page()