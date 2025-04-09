import streamlit as st
import librosa
import soundfile as sf
import os
import numpy as np

from model_wrapper import deepfilter_denoise
from utils import plot_spectrogram, estimate_noise_level

st.set_page_config(page_title="DeepFilterNet Denoising", layout="wide")
st.title("🎧 DeepFilterNet -  Noise Cancellation")

uploaded_file = st.file_uploader("📂 Upload a noisy WAV audio file", type=["wav"])

if uploaded_file:
    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Sample rate: {sr}, Duration: {len(audio) / sr:.2f} sec")

    # 🔄 Adaptive Mode Toggle
    adaptive_mode = st.checkbox("🔄 Enable Adaptive Mode", value=True)

    # 🧠 Estimate Noise Level
    noise_level = estimate_noise_level(audio, sr)
    if adaptive_mode:
        st.write(f"🧠 Estimated Noise Level: `{noise_level:.2f}`")
        if noise_level < 0.3:
            st.success("Noise Level: Low")
        elif noise_level < 0.6:
            st.warning("Noise Level: Medium")
        else:
            st.error("Noise Level: High")

    # 🎵 Show Original Spectrogram
    st.subheader("🎵 Original Spectrogram")
    st.pyplot(plot_spectrogram(audio, sr, title="Original Spectrogram"))

    # 🧠 Run DeepFilterNet
    with st.spinner("🔧 Running DeepFilterNet..."):
        denoised_audio, new_sr = deepfilter_denoise(audio, sr)

    # 🔧 Apply Adaptive Gain Control
    if adaptive_mode:
        post_gain = max(0.3, 1.0 - noise_level)
        denoised_audio *= post_gain
        st.caption(f"🔧 Adaptive post-filter applied with gain = `{post_gain:.2f}`")

    # 💾 Save and Play Denoised Audio
    output_path = os.path.join("assets", "denoised_output.wav")
    sf.write(output_path, denoised_audio, new_sr)

    st.subheader("🔈 Denoised Audio")
    st.audio(output_path, format="audio/wav")

    # 🎵 Show Denoised Spectrogram
    st.subheader("🎵 Denoised Spectrogram")
    st.pyplot(plot_spectrogram(denoised_audio, new_sr, title="Denoised Spectrogram"))
