import torch
import torch.nn.functional as F
import torchaudio

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
        self.n_fft = win_length if n_fft is None else n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            power=1,
            center=False,
        )

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mel_channels,
            sample_rate=self.sampling_rate,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_stft=self.n_fft // 2 + 1,
            mel_scale="htk",
        )

    def forward(self, audio, center=True):
        if center:
            pad_left = self.n_fft // 2
            pad_right = (self.n_fft + 1) // 2
            audio = F.pad(audio, (pad_left, pad_right))

        spectrogram = self.spectrogram(audio)
        mel_output = self.mel_scale(spectrogram)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class MelSpecExtractor(torch.nn.Module):
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
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.melspec_transform = MelSpectrogram(
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
        return self.melspec_transform(waveform.unsqueeze(0))
