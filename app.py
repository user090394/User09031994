import streamlit as st
import librosa
import numpy as np
import random
from pydub import AudioSegment
import io
import tempfile
import os

# Оптимізована функція
def process_audio_web(uploaded_file, glitch_chance, intensity, grain_size):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(uploaded_file.getbuffer())
        tmp_in_path = tmp_in.name

    try:
        # Завантажуємо з меншою частотою дискретизації для економії пам'яті
        y, sr = librosa.load(tmp_in_path, sr=22050) 
        audio = AudioSegment.from_file(tmp_in_path)
        
        # Детекція ритму
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onsets_ms = (librosa.frames_to_time(onset_frames, sr=sr) * 1000).astype(int)
        onsets_ms = np.append(onsets_ms, len(audio))
        
        glitched_audio = AudioSegment.empty()
        
        # Обробляємо частинами
        for i in range(len(onsets_ms) - 1):
            start, end = onsets_ms[i], onsets_ms[i+1]
            chunk = audio[start:end]
            if len(chunk) < 10: continue

            if random.random() * 100 < glitch_chance:
                r = random.random() * 100
                if r < (intensity * 0.4):
                    grain = chunk[:grain_size]
                    glitched_audio += (grain * 15)[:len(chunk)*2] # Обмеження довжини
                elif r < (intensity * 0.7):
                    glitched_audio += (chunk[:30] * 5)
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

st.title("🎹 AFX Machine")

# Додаємо обмеження на розмір у самому інтерфейсі
uploaded_file = st.file_uploader("Завантаж файл (до 20МБ для стабільності)", type=["wav", "mp3"])

if uploaded_file:
    # Показуємо плеєр тільки якщо файл не занадто великий
    st.audio(uploaded_file)
    
    gc = st.sidebar.slider("Glitch Chance %", 0, 100, 30)
    gi = st.sidebar.slider("Intensity %", 0, 100, 50)
    gs = st.sidebar.slider("Grain ms", 5, 100, 15)

    if st.button("ГЛІЧУВАТИ"):
        with st.spinner("Processing..."):
            try:
                res = process_audio_web(uploaded_file, gc, gi, gs)
                st.audio(res)
                st.download_button("СКАЧАТИ", res.getvalue(), "glitch.wav")
            except Exception as e:
                st.error(f"Помилка обробки: {e}")
