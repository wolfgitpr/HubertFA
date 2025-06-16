import os
import pathlib

import click
import numpy as np
import torch
import tqdm

from config_utils import load_yaml
from tools.encoder import UnitsEncoder
from tools.load_wav import load_wav


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="configs/binarize_config.yaml",
    show_default=True,
    help="binarize config path",
)
def pretreatment_hubert(config: str):
    config = load_yaml(config)

    datasets_config = config["datasets_config"]
    assert isinstance(datasets_config, list), f"{datasets_config} is not a list."

    datasets = []
    for dataset_path in datasets_config:
        if os.path.exists(dataset_path):
            datasets.extend(load_yaml(dataset_path)["datasets"])

    sample_rate = config['melspec_config']["sample_rate"]
    max_length = config['max_length']
    hop_size = config['melspec_config']["hop_length"]
    hubert_channel = config['hubert_config']["channel"]

    all_wav_paths = []

    for dataset in datasets:
        raw_data_dir = pathlib.Path(dataset["raw_data_dir"])
        all_wav_paths.extend([i for i in raw_data_dir.rglob("*.wav")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unitsEncoder = UnitsEncoder(config['hubert_config'], config['melspec_config'], device=device)

    for wav_path in tqdm.tqdm(all_wav_paths):
        try:
            if not os.path.exists(wav_path):
                print(f"Skipping {wav_path}, because it doesn't exist")

            npy_path = pathlib.Path(wav_path).with_suffix(".npy")
            if not os.path.exists(npy_path):
                waveform = load_wav(wav_path, device, sample_rate)  # (L,)

                wav_length = len(waveform) / sample_rate  # seconds
                if wav_length > max_length:
                    print(
                        f"Item {wav_path} has a length of {wav_length}s, which is too long, skip it."
                    )

                # units encode
                units = unitsEncoder.forward(waveform.unsqueeze(0), sample_rate, hop_size)  # [B, T, C]

                B, T, C = units.shape
                assert C == hubert_channel, f"Item {wav_path} has unexpect channel of {C}, which should be {hubert_channel}."

                np.save(npy_path, units.float().cpu().numpy())
        except Exception as e:
            print(f"error: {wav_path}: {str(e)}")


if __name__ == "__main__":
    pretreatment_hubert()
