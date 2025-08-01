import os.path
import pathlib

import torch
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class UnitsEncoder(torch.nn.Module):
    def __init__(self, hubert_config: dict, mel_config: dict, encoder_ckpt=None, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.mel_config = mel_config

        self.encoder = hubert_config.get("encoder", "cnhubert")
        if encoder_ckpt is None:
            encoder_ckpt = hubert_config.get("model_path", None)

        is_loaded_encoder = False
        if self.encoder == 'cnhubert':
            self.model = Audio2CNHubert(encoder_ckpt, device=device)
            is_loaded_encoder = True
        assert is_loaded_encoder, f" [x] Unknown units encoder: {self.encoder}"

        self.resample_kernel = {}
        self.encoder_sample_rate = hubert_config.get("sample_rate", 16000)
        self.encoder_hop_size = hubert_config.get("hop_size", 320)

    def forward(self,
                audio,  # B, T
                sample_rate,
                hop_size):
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))  # [B, T]
        units = self.model(audio_res)  # [B, T, C]

        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
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
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            path, local_files_only=True)

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            input_values = self.feature_extractor(audio, return_tensors="pt",
                                                  sampling_rate=16000).input_values.to(audio.device).squeeze(1)
            return self.model(input_values)["last_hidden_state"]  # [1, T, C]
