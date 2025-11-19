import os.path
import pathlib
import random

import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import HubertModel


class UnitsEncoder(torch.nn.Module):
    def __init__(self, hubert_config: dict, mel_config: dict, encoder_ckpt=None, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.mel_config = mel_config

        self.encoder = hubert_config["encoder"]
        if encoder_ckpt is None:
            encoder_ckpt = hubert_config.get("model_path", None)

        is_loaded_encoder = False
        if self.encoder == 'cn_hubert':
            self.model = Audio2CNHubert(encoder_ckpt, device=device)
            is_loaded_encoder = True
        assert is_loaded_encoder, f" [x] Unknown units encoder: {self.encoder}"

        self.resample_kernel = {}
        self.encoder_sample_rate: int = hubert_config["sample_rate"]
        self.encoder_hop_size: int = hubert_config["hop_size"]

    def forward(self,
                audio,  # B, T
                sample_rate,
                hop_size,
                aug=False,
                aug_args=None,
                ):
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_resample = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_resample = self.resample_kernel[key_str](audio)

        # encode
        if audio_resample.size(-1) < 400:
            audio_resample = torch.nn.functional.pad(audio, (0, 400 - audio_resample.size(-1)))  # [B, T]

        if aug and aug_args['random_pitch_shifting']['enabled']:
            key_shift_min, key_shift_max = aug_args['random_pitch_shifting']['range']
            audio_versions = [audio_resample]
            for _ in range(aug_args['random_pitch_shifting']['num']):
                rand = random.uniform(-1, 1)
                key_shift = key_shift_min * abs(rand) if rand < 0 else key_shift_max * rand
                pitch_shifted = torchaudio.functional.pitch_shift(audio_resample[0], self.encoder_sample_rate,
                                                                  key_shift)
                if pitch_shifted.dim() == 1:
                    pitch_shifted = pitch_shifted.unsqueeze(0)
                audio_versions.append(pitch_shifted)
            audio_resample = torch.cat(audio_versions, dim=0)

        units = self.model(audio_resample)  # [B, T, C]

        if aug and aug_args['blank_padding']['enabled']:
            padding_min, padding_max = aug_args['blank_padding']['range']
            audio_length = len(audio_resample[0])
            audio_padding = torch.zeros(
                (aug_args['blank_padding']['num'], audio_length + padding_max * self.encoder_sample_rate),
                device=self.device)
            padding_lengths = []
            for i in range(aug_args['blank_padding']['num']):
                padding_samples = random.randint(padding_min, padding_max) * self.encoder_sample_rate
                padding_lengths.append(padding_samples)
                audio_padding[i, padding_samples:audio_length + padding_samples] = audio_resample[0]

            units_padding = self.model(audio_padding)

            units_versions = [units]
            for i in range(len(units_padding)):
                padding_frames = int(padding_lengths[i] / self.encoder_hop_size)
                units_versions.append(units_padding[i:i + 1, padding_frames:padding_frames + len(units[0])])
            units = torch.cat(units_versions, dim=0)

        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1,
                                     index.unsqueeze(0).unsqueeze(-1).repeat([units.size(0), 1, units.size(-1)]))
        return units_aligned  # [B, T, C]


class Audio2CNHubert(torch.nn.Module):
    def __init__(self, path, device='cpu'):
        super().__init__()
        print(' [Encoder Model] Chinese Hubert')
        print(' [Loading] ' + path)
        if os.path.isfile(path):
            path = str(pathlib.Path(path).parent)
        self.model = HubertModel.from_pretrained(path, local_files_only=True).to(device)
        self.model.eval()

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            return self.model(audio)["last_hidden_state"]  # [1, T, C]
