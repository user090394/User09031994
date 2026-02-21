import streamlit as st
import librosa
import numpy as np
import random
from pydub import AudioSegment
import io
import tempfile
import os

def process_audio_web(uploaded_file, intensity, grain_size):
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

            r = random.random() * 100
            if r < (intensity * 0.3): # Granular Stretch
                grain = chunk[:grain_size].fade_out(2)
                glitched_audio += grain * 20
            elif r < (intensity * 0.6): # Stutter
                glitched_audio += chunk[:25] * 8
            elif r < intensity: # Reverse
                glitched_audio += chunk.reverse()
            else:
                glitched_audio += chunk
        
        out_buffer = io.BytesIO()
        glitched_audio.export(out_buffer, format="wav")
        return out_buffer

    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

# UI Налаштування
st.set_page_config(page_title="AFX Deconstructor", layout="centered")
st.title("🎹 AFX Deconstructor")
st.markdown("Деконструюй свої треки прямо тут.")

uploaded_file = st.file_uploader("Завантаж MP3 або WAV", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    
    intensity = st.sidebar.slider("Chaos Intensity (%)", 0, 100, 40)
    grain = st.sidebar.slider("Grain Size (ms)", 5, 100, 15)

    if st.button("ГЛІЧУВАТИ", use_container_width=True):
        with st.spinner("Обробка..."):
            res = process_audio_web(uploaded_file, intensity, grain)
            st.audio(res, format='audio/wav')
            st.download_button("СКАЧАТИ РЕЗУЛЬТАТ", res.getvalue(), "glitch_track.wav")
