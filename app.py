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

            # Визначаємо, чи буде цей шматочок глічовим
            if random.random() * 100 < glitch_chance:
                # Якщо ТАК — застосовуємо один з ефектів залежно від intensity
                r = random.random() * 100
                if r < (intensity * 0.4): # Granular
                    grain = chunk[:grain_size].fade_out(2)
                    glitched_audio += grain * 20
                elif r < (intensity * 0.7): # Stutter
                    glitched_audio += chunk[:25] * 8
                else: # Reverse
                    glitched_audio += chunk.reverse()
            else:
                # Якщо НІ — залишаємо звук чистим
                glitched_audio += chunk
        
        out_buffer = io.BytesIO()
        glitched_audio.export(out_buffer, format="wav")
        return out_buffer

    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

# --- UI ---
st.set_page_config(page_title="AFX Deconstructor v5.0", layout="centered")
st.title("🎹 AFX Deconstructor")
st.markdown("Налаштуй частоту та силу цифрового хаосу.")

uploaded_file = st.file_uploader("Завантаж трек", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    
    st.sidebar.header("Налаштування глічу")
    
    # НОВИЙ ПОВЗУНОК: Як часто робити гліч
    glitch_chance = st.sidebar.slider("Частота глічів (Glitch Chance %)", 0, 100, 30)
    
    # Старі повзунки для характеру глічу
    intensity = st.sidebar.slider("Сила ефектів (Intensity %)", 0, 100, 50)
    grain = st.sidebar.slider("Розмір гранул (Grain ms)", 5, 100, 15)

    if st.button("ГЛІЧУВАТИ", use_container_width=True):
        with st.spinner("Алгоритми Афекса працюють..."):
            res = process_audio_web(uploaded_file, glitch_chance, intensity, grain)
            st.audio(res, format='audio/wav')
            st.download_button("СКАЧАТИ РЕЗУЛЬТАТ", res.getvalue(), "afx_glitch_v5.wav")
