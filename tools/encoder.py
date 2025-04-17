import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.model import ModelDimensions, Whisper

from networks.hubert.model import HubertSoft
from tools.get_melspec import MelSpecExtractor


class UnitsEncoder(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.mel_config = mel_config

        self.encoder = hubert_config.get("encoder", "mel")
        encoder_ckpt = hubert_config.get("model_path", None)

        is_loaded_encoder = False
        if self.encoder == "mel":
            self.model = MelSpecExtractor(**self.mel_config, device=device)
            is_loaded_encoder = True
        if self.encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if self.encoder == 'cnhubert':
            self.model = Audio2CNHubert(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if self.encoder == 'whisper-ppg':
            self.model = Audio2Whisper(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if self.encoder == 'hubertsofttta2x':
            self.model = Audio2HubertSoftTTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        assert is_loaded_encoder, f" [x] Unknown units encoder: {self.encoder}"

        self.resample_kernel = {}
        self.encoder_sample_rate = hubert_config.get("sample_rate", 16000)
        self.encoder_hop_size = hubert_config.get("hop_size", 320)

    def forward(self,
                audio,  # B, T
                sample_rate,
                hop_size):
        if self.encoder == "mel":
            return self.model(audio.squeeze(0))
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
        return units_aligned.transpose(1, 2)  # [B, C, T]


class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft().to(device)
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)["hubert"]
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            units = self.hubert.units(audio.unsqueeze(1))
            return units  # [1, T, C]


class Audio2CNHubert(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] Chinese Hubert')
        print(' [Loading] ' + path)
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


class Audio2Whisper(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] Whisper')
        print(' [Loading] ' + path)

        self.dev = device
        checkpoint = torch.load(path, map_location=self.dev)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def forward(self,
                audio):  # B, T
        audln = audio.shape[1]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            return ppg[None, :, :]  # [1, T, C]


class Audio2HubertSoftTTA2X:
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Hubert Soft with TTA 2X')
        print(' [Loading] ' + path)
        self.hubert = HubertSoft()
        checkpoint = torch.load(path)["hubert"]
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(device)
        self.hubert.eval()

    def __call__(self, audio):
        # audio: [B, T]
        with torch.no_grad():
            feats = self.hubert.units(audio.unsqueeze(1))
            padded_audio = F.pad(audio, (160, 0))  # [B, T + pad_amount]
            feats2 = self.hubert.units(padded_audio.unsqueeze(1))
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
        units = feats_tta  # .transpose(2, 1)
        return units  # [1, T, B]
