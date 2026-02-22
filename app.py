import streamlit as st
import librosa
import numpy as np
import random
from pydub import AudioSegment
import io
import tempfile
import os

def process_audio_web(uploaded_file, glitch_chance, intensity, grain_size):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(uploaded_file.getbuffer())
        tmp_in_path = tmp_in.name

    try:
        y, sr = librosa.load(tmp_in_path, sr=None)
        audio = AudioSegment.from_file(tmp_in_path)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onsets_ms = (librosa.frames_to_time(onset_frames, sr=sr) * 1000).astype(int)
        onsets_ms = np.append(onsets_ms, len(audio))
        glitched_audio = AudioSegment.empty()
        
        for i in range(len(onsets_ms) - 1):
            start, end = onsets_ms[i], onsets_ms[i+1]
            chunk = audio[start:end]
            if len(chunk) == 0: continue

            if random.random() * 100 < glitch_chance:
                r = random.random() * 100
                if r < (intensity * 0.4):
                    grain = chunk[:grain_size].fade_out(2)
                    glitched_audio += grain * 20
                elif r < (intensity * 0.7):
                    glitched_audio += chunk[:25] * 8
                else:
                    glitched_audio += chunk.reverse()
            else:
                glitched_audio += chunk
        
        out_buffer = io.BytesIO()
        glitched_audio.export(out_buffer, format="wav")
        return out_buffer
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

st.set_page_config(page_title="AFX Deconstructor")
st.title("🎹 AFX Deconstructor")

uploaded_file = st.file_uploader("Завантаж трек", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    g_chance = st.sidebar.slider("Glitch Chance %", 0, 100, 30)
    g_intensity = st.sidebar.slider("Intensity %", 0, 100, 50)
    g_grain = st.sidebar.slider("Grain ms", 5, 100, 15)

    if st.button("ГЛІЧУВАТИ"):
        with st.spinner("Processing..."):
            res = process_audio_web(uploaded_file, g_chance, g_intensity, g_grain)
            st.audio(res, format='audio/wav')
            st.download_button("СКАЧАТИ", res.getvalue(), "glitch.wav")
