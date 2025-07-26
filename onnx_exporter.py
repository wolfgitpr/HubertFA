import os.path
import pathlib
import shutil
from pathlib import Path

import click
import onnx
import onnxsim
import torch
import torchaudio
import yaml
from transformers import HubertModel

from networks.task.forced_alignment import LitForcedAlignmentTask
from tools.config_utils import load_yaml

ONNX_EXPORT_VERSION = 2


class UnitsAligner(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.hop_size = mel_config["hop_length"]
        self.sample_rate = mel_config["sample_rate"]
        self.encoder_sample_rate = hubert_config.get("sample_rate", 16000)
        self.encoder_hop_size = hubert_config.get("hop_size", 320)

    def forward(self,
                units,  # [B, T, C]
                n_frames):
        # alignment
        ratio = (self.hop_size / self.sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned  # [B, T, C]


class Resampler(torch.nn.Module):
    def __init__(self, hubert_config, mel_config, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hop_size = mel_config["hop_length"]
        self.resampler = torchaudio.transforms.Resample(mel_config["sample_rate"], hubert_config["sample_rate"]).to(
            device)

    def forward(self,
                waveform,
                ):
        n_frames = waveform.size(-1) // self.hop_size + 1
        return self.resampler(waveform), n_frames


def change_input_node_name(model, input_names):
    for i, _input in enumerate(model.graph.input):
        input_name = input_names[i]
        for node in model.graph.node:
            for j, name in enumerate(node.input):
                if name == _input.name:
                    node.input[j] = input_name
        _input.name = input_name


def change_output_node_name(model, output_names):
    for i, output in enumerate(model.graph.output):
        output_name = output_names[i]
        for node in model.graph.node:
            for j, name in enumerate(node.output):
                if name == output.name:
                    node.output[j] = output_name
        output.name = output_name


def export_hubert_encoder(_encoder_path, _hubert_config, _melspec_config, device="cpu"):
    encoder_folder = pathlib.Path(_encoder_path).parent

    # export resampler
    resampler_path = Path(encoder_folder) / "resampler.onnx"
    resampler_model = Resampler(_hubert_config, _melspec_config, device)
    waveform = torch.randn((1, 44100), dtype=torch.float32, device=device)
    torch.onnx.export(
        resampler_model,
        waveform,
        str(resampler_path),
        input_names=['waveform'],
        output_names=['waveform_16k', 'n_frames'],
        dynamic_axes={
            'waveform': {1: 'n_samples'},
            'waveform_16k': {1: 'n_samples'},
        }
    )

    # export hubert
    hubert_path = _hubert_config["model_path"]
    model = HubertModel.from_pretrained(hubert_path).to(device).eval()
    hubert_onnx_path = Path(encoder_folder) / "bert_model.onnx"
    dummy_input = torch.randn(1, 16000).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(hubert_onnx_path),
        opset_version=17,
        input_names=["waveform_16k"],
        output_names=["units"],
        dynamic_axes={
            "waveform_16k": {1: "n_samples"},
            "units": {1: "n_samples"}
        }
    )

    # export aligner
    aligner_path = Path(encoder_folder) / "hubert_aligner.onnx"
    aligner = UnitsAligner(_hubert_config, _melspec_config, device)
    units = torch.randn((1, _hubert_config["channel"], 1024), dtype=torch.float32, device=device)
    n_frames = 500
    torch.onnx.export(
        aligner,
        (units, n_frames),
        str(aligner_path),
        opset_version=17,
        input_names=["units", "n_frames"],
        output_names=["input_feature"],
        dynamic_axes={
            "units": {1: "n_frames"},
            "input_feature": {1: "n_frames"},
        }
    )

    # merge model
    resampler = onnx.load(resampler_path)
    resampler = onnx.compose.add_prefix(resampler, prefix="resampler/")

    hubert = onnx.load(hubert_onnx_path)
    hubert = onnx.compose.add_prefix(hubert, prefix="hubert/")

    aligner = onnx.load(aligner_path)
    aligner = onnx.compose.add_prefix(aligner, prefix="aligner/")

    intermediate = onnx.compose.merge_models(
        resampler, hubert,
        io_map=[("resampler/waveform_16k", "hubert/waveform_16k")]
    )

    final_model = onnx.compose.merge_models(
        intermediate, aligner,
        io_map=[
            ("hubert/units", "aligner/units"),
            ("resampler/n_frames", "aligner/n_frames")
        ]
    )

    onnx.save(final_model, _encoder_path)

    onnx_model, check = onnxsim.simplify(_encoder_path, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    change_input_node_name(onnx_model, ["waveform"])
    change_output_node_name(onnx_model, ["input_feature"])
    onnx.save(onnx_model, _encoder_path)
    print(f"Merged encoder saved to: {_encoder_path}")

    os.remove(resampler_path)
    os.remove(aligner_path)
    os.remove(hubert_onnx_path)


def export_fa_model(_fa_path, ckpt_path, _hubert_config, _melspec_config, device="cpu"):
    ckpt_folder = pathlib.Path(ckpt_path).parent
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt_path, strict=False).to(device)

    # (B, T, C)
    input_feature = torch.randn((1, 1000, _hubert_config.get("channel", 128)), dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_feature,
            _fa_path,
            input_names=['input_feature'],
            output_names=['ph_frame_logits', 'ph_edge_logits', 'ctc_logits', 'cvnt_logits'],
            dynamic_axes={
                'input_feature': {1: 'n_samples'},
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


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--onnx_folder', required=True, metavar='DIR', help='Path to the onnx')
def export(ckpt_path, onnx_folder):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    ckpt_folder = pathlib.Path(ckpt_path).parent
    onnx_folder = pathlib.Path(onnx_folder)
    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert os.path.exists(ckpt_folder / "config.yaml"), f"Checkpoint folder does not exist: config.yaml"

    config = load_yaml(ckpt_folder / "config.yaml")

    hubert_config = config["hubert_config"]
    melspec_config = config["melspec_config"]
    encoder_name = hubert_config["encoder"]

    os.makedirs(onnx_folder, exist_ok=True)
    encoder_path = str(onnx_folder / f"{encoder_name}-{hubert_config['channel']}.onnx")
    onnx_path = str(onnx_folder / "model.onnx")

    assert hubert_config["encoder"] in ["cnhubert"], f"{hubert_config['encoder']} must be 'cnhubert'"
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."

    export_hubert_encoder(encoder_path, hubert_config, melspec_config)

    export_fa_model(onnx_path, ckpt_path, hubert_config, melspec_config)


if __name__ == '__main__':
    export()
