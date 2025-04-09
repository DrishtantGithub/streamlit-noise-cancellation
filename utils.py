# utils.py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(audio, sr, title="Spectrogram"):
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    ax.label_outer()
    return fig


def estimate_noise_level(audio: np.ndarray, sr: int) -> float:
    """Estimate noise level from first 0.5 seconds of audio."""
    sample_count = int(0.5 * sr)
    noise_sample = audio[:sample_count]
    noise_level = np.mean(np.abs(noise_sample))
    return min(1.0, noise_level * 10)  # normalize to [0, 1]


