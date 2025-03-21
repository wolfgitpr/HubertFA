import torch
import numpy as np
import torch.nn.functional as F
from librosa.filters import mel

melspec_transform = None


class MelSpectrogram(torch.nn.Module):
    def __init__(
            self,
            n_mel_channels,
            sampling_rate,
            win_length,
            hop_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + '_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
        if center:
            pad_left = n_fft_new // 2
            pad_right = (n_fft_new + 1) // 2
            audio = F.pad(audio, (pad_left, pad_right))

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=False,
            return_complex=True
        )
        magnitude = fft.abs()

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class MelSpecExtractor:
    def __init__(
            self,
            n_mels,
            sample_rate,
            win_length,
            hop_length,
            n_fft,
            fmin,
            fmax,
            clamp,
            device=None,
            scale_factor=None,
    ):
        global melspec_transform
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if melspec_transform is None:
            melspec_transform = MelSpectrogram(
                n_mel_channels=n_mels,
                sampling_rate=sample_rate,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                mel_fmin=fmin,
                mel_fmax=fmax,
                clamp=clamp,
            ).to(device)

    def __call__(self, waveform, key_shift=0):
        return melspec_transform(waveform.unsqueeze(0), key_shift).squeeze(0)
