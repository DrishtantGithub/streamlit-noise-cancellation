# model_wrapper.py

from df.enhance import enhance, init_df
import numpy as np
import torch
import torchaudio

# Load the DeepFilterNet model once
model, df_state, _ = init_df()

def deepfilter_denoise(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    # Convert to torch Tensor with shape [1, N]
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, N]

    # Resample to 48000 Hz if needed
    if sr != 48000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 48000)
        sr = 48000

    # Run the model: input must be shape [1, N]
    enhanced_tensor = enhance(model, df_state, audio_tensor)

    # Convert output back to NumPy
    return enhanced_tensor.squeeze(0).numpy(), sr
