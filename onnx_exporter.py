import os.path
import pathlib
import shutil

import click
import onnx
import onnxsim
import torch
import yaml
from librosa.filters import mel
from torch import nn

from networks.task.forced_alignment import LitForcedAlignmentTask


class MelSpectrogram_ONNX(nn.Module):
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

    def forward(self, audio, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            center=center,
            return_complex=False
        )
        magnitude = torch.sqrt(torch.sum(fft ** 2, dim=-1))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--onnx_foler', required=True, metavar='DIR', help='Path to the onnx')
def export(ckpt_path, onnx_foler):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    ckpt_folder = pathlib.Path(ckpt_path).parent
    onnx_foler = pathlib.Path(onnx_foler)
    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert os.path.exists(ckpt_folder / "config.yaml"), f"Checkpoint folder does not exist: config.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ckpt_folder / "config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())

    hubert_config = config["hubert_config"]
    melspec_config = config["melspec_config"]

    os.makedirs(onnx_foler, exist_ok=True)
    encoder_path = str(onnx_foler / "encoder.onnx")
    onnx_path = str(onnx_foler / "model.onnx")

    assert hubert_config["encoder"] == "mel", f"{hubert_config['encoder']} must be 'mel'"
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."

    encoder = MelSpectrogram_ONNX(
        melspec_config["n_mels"], melspec_config["sample_rate"], melspec_config["win_length"],
        melspec_config["hop_length"], melspec_config["n_fft"], melspec_config["fmin"], melspec_config["fmax"]
    )
    waveform = torch.randn((1, 44100), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            waveform,
            encoder_path,
            input_names=['waveform'],
            output_names=['input_feature'],
            dynamic_axes={
                'waveform': {1: 'n_samples'},
                'input_feature': {1: 'n_samples'},
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(encoder_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, encoder_path)
        print(f'Encoder Model saved to: {encoder_path}')

    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt_path, strict=False).to(device)

    # (B, T, C)
    input_feature = torch.randn((1, 1000, hubert_config.get("channel", 128)), dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_feature,
            onnx_path,
            input_names=['input_feature'],
            output_names=['ph_frame_logits', 'ph_edge_logits', 'ctc_logits'],
            dynamic_axes={
                'input_feature': {1: 'n_samples'},
                'ph_frame_logits': {1: 'n_samples'},  # (T, vocab_size)
                'ph_edge_logits': {1: 'n_samples'},  # (T)
                'ctc_logits': {1: 'n_samples'},  # (T, vocab_size)
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(onnx_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, onnx_path)
        print(f'Predict Model saved to: {onnx_path}')

    src_files = list(ckpt_folder.glob("*.yaml")) + list(ckpt_folder.glob("*.txt"))

    for i in src_files:
        shutil.copy(str(i), pathlib.Path(onnx_path).parent)

    with open(pathlib.Path(onnx_path).parent / 'hparams.yaml', 'w') as f:
        yaml.dump(model.hparams, f, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    export()
