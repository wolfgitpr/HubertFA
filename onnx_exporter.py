import os.path
import pathlib
import shutil

import click
import onnx
import onnxsim
import torch
import torchaudio
import yaml
from transformers import HubertModel

from networks.task.forced_alignment import LitForcedAlignmentTask
from tools.binarize_util import PowerCurveComputer
from tools.config_utils import load_yaml

ONNX_EXPORT_VERSION = 3


class Resampler(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hop_size = mel_config["hop_size"]
        self.resampler = torchaudio.transforms.Resample(mel_config["sample_rate"], hubert_config["sample_rate"]).to(
            device)

    def forward(self,
                waveform,
                ):
        n_frames = waveform.size(-1) // self.hop_size + 1
        return self.resampler(waveform), n_frames


class UnitsAligner(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.ratio = (mel_config["hop_size"] / mel_config["sample_rate"]) / (
                hubert_config["hop_size"] / hubert_config["sample_rate"])

    def forward(self,
                units,  # [B, T, C]
                n_frames):
        # alignment
        index = torch.clamp(torch.round(self.ratio * torch.arange(n_frames).to(self.device)).long(),
                            max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned  # [B, T, C]


class Encoder(torch.nn.Module):
    def __init__(self, hubert_config, melspec_config, hubert_path=None, device='cpu'):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hubert_path is None:
            hubert_path = hubert_config["model_path"]
        self.resampler = Resampler(hubert_config, melspec_config, device)
        self.curves_computer = PowerCurveComputer(melspec_config["window_size"], melspec_config["hop_size"])
        self.hubert = HubertModel.from_pretrained(hubert_path).to(device).eval()
        self.aligner = UnitsAligner(hubert_config, melspec_config, device)

    def forward(self, waveform):
        waveform_16k, n_frames = self.resampler(waveform)
        curves = self.curves_computer(waveform.squeeze(0), n_frames).unsqueeze(0).unsqueeze(0)
        units = self.hubert(waveform_16k)["last_hidden_state"]
        units_aligned = self.aligner(units, n_frames)
        return units_aligned, curves


def export_encoder(encoder_folder, _hubert_config, _mel_config, hubert_path=None, device='cpu'):
    if hubert_path is None:
        hubert_path = _hubert_config["model_path"]
    encoder = Encoder(_hubert_config, _mel_config, hubert_path)
    encoder_onnx_path = str(pathlib.Path(encoder_folder) / "encoder.onnx")

    waveform = torch.randn(1, 22050).to(device)
    torch.onnx.export(
        encoder,
        waveform,
        str(encoder_onnx_path),
        opset_version=17,
        input_names=["waveform"],
        output_names=["units_aligned", "curves_out"],
        dynamic_axes={
            "waveform": {1: "n_samples"},
            "units_aligned": {1: "n_samples"},
            "curves_out": {2: "n_frames"},
        }
    )
    onnx_model, check = onnxsim.simplify(encoder_onnx_path, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(onnx_model, encoder_onnx_path)
    print(f'Encoder Model saved to: {encoder_onnx_path}')
    return encoder_onnx_path


def export_fa_model(_fa_path, ckpt_path, _hubert_config, _melspec_config, device="cpu"):
    ckpt_folder = pathlib.Path(ckpt_path).parent
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt_path, strict=False).to(device)

    n_frames = 1000
    input_feature = torch.randn((1, n_frames, _hubert_config["channel"]), dtype=torch.float32,
                                device=device)  # (B, T, C)
    curves = torch.randn((1, 1, n_frames), dtype=torch.float32, device=device)  # (B, 1, T)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_feature, curves),
            _fa_path,
            input_names=['input_feature', 'curves_in'],
            output_names=['ph_frame_logits', 'ph_edge_logits', 'ctc_logits', 'cvnt_logits'],
            dynamic_axes={
                'input_feature': {1: 'n_samples'},
                'curves_in': {2: 'n_samples'},
                'ph_frame_logits': {1: 'n_samples'},  # (T, vocab_size)
                'ph_edge_logits': {1: 'n_samples'},  # (T)
                'ctc_logits': {1: 'n_samples'},  # (T, vocab_size)
                'cvnt_logits': {1: 'n_samples'},  # (N, T)
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(_fa_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, _fa_path)
        print(f'Predict Model saved to: {_fa_path}')

        src_files = list(ckpt_folder.glob("*.yaml")) + list(ckpt_folder.glob("*.txt"))

        for i in src_files:
            shutil.copy(str(i), pathlib.Path(_fa_path).parent)

        with open(pathlib.Path(_fa_path).parent / 'hparams.yaml', 'w') as f:
            yaml.dump(model.hparams, f, default_flow_style=False, allow_unicode=True)

        with open(pathlib.Path(_fa_path).parent / 'VERSION', 'w') as f:
            f.write(str(ONNX_EXPORT_VERSION))
    return _fa_path


def merge_model(merged_path, encoder_path, fa_path):
    final_model = onnx.compose.merge_models(
        onnx.load(encoder_path), onnx.load(fa_path),
        io_map=[
            ("units_aligned", "input_feature"),
            ("curves_out", "curves_in")
        ]
    )
    onnx.save(final_model, merged_path)

    onnx_model, check = onnxsim.simplify(merged_path, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    onnx.save(onnx_model, merged_path)
    print(f"Merged encoder saved to: {merged_path}")

    os.remove(encoder_path)
    os.remove(fa_path)


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option("--hubert_path", "-h", default=None, type=str, help="path to the encoder model")
@click.option('--onnx_folder', required=True, metavar='DIR', help='Path to the onnx')
def export(ckpt_path, hubert_path, onnx_folder):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    ckpt_folder = pathlib.Path(ckpt_path).parent
    onnx_folder = pathlib.Path(onnx_folder)
    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert os.path.exists(ckpt_folder / "config.yaml"), f"Checkpoint folder does not exist: config.yaml"

    os.makedirs(onnx_folder, exist_ok=True)
    onnx_path = str(onnx_folder / "model.onnx")
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."

    config = load_yaml(ckpt_folder / "config.yaml")

    encoder_path = export_encoder(onnx_folder, config["hubert_config"], config["melspec_config"],
                                  hubert_path=hubert_path)
    fa_path = export_fa_model(onnx_path, ckpt_path, config["hubert_config"], config["melspec_config"])

    merged_path = str(onnx_folder / "model.onnx")
    merge_model(merged_path, encoder_path, fa_path)


if __name__ == '__main__':
    export()
