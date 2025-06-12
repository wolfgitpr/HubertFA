import os
import pathlib

import click
import lightning as pl
import torch

import networks.g2p
from tools.config_utils import check_configs, load_yaml
from tools.encoder import UnitsEncoder
from tools.export_tool import Exporter
from tools.post_processing import post_processing
from train import LitForcedAlignmentTask


@click.command()
@click.option(
    "--ckpt",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the checkpoint",
)
@click.option(
    "--encoder", "-e", default=None, type=str, help="path to the encoder model"
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class"
)
@click.option(
    "--save_confidence",
    "-sc",
    is_flag=True,
    default=False,
    show_default=True,
    help="save confidence.csv",
)
@click.option(
    "--language",
    "-l",
    default="zh",
    type=str,
    help="language of dictionary.(exp. zh ja en yue)",
)
@click.option(
    "--dictionary",
    "-d",
    default=None,
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def main(
        ckpt,
        encoder,
        folder,
        g2p,
        save_confidence,
        **kwargs,
):
    model_dir = pathlib.Path(ckpt).parent
    check_configs(model_dir)

    if "Dictionary" in g2p:
        if kwargs["dictionary"] is None:
            vocab = load_yaml(model_dir / "vocab.yaml")
            dictionary_path = model_dir / vocab["dictionaries"].get(kwargs["language"], "")
            kwargs["dictionary"] = dictionary_path
        assert os.path.exists(kwargs["dictionary"]), f"{pathlib.Path(kwargs['dictionary']).absolute()} does not exist"

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    if encoder is not None:
        model.unitsEncoder = UnitsEncoder(model.hubert_config, model.melspec_config, encoder, model.device)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions, log = post_processing(predictions)
    if log:
        print("error:", "\n".join(log))

    exporter = Exporter(predictions)
    exporter.export(['textgrid'] if not save_confidence else ['textgrid', 'confidence'])

    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
