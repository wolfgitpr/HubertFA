import pathlib

import click
import lightning as pl
import torch

import networks.g2p
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
    "-d",
    default="zh",
    type=str,
    help="language of dictionary.(exp. zh ja en yue)",
)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def main(
        ckpt,
        folder,
        g2p,
        save_confidence,
        **kwargs,
):
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)

    grapheme_to_phoneme.set_in_format('lab')
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions, log = post_processing(predictions)
    exporter = Exporter(predictions, log)

    out_formats = ['textgrid']
    if save_confidence:
        out_formats.append('confidence')

    exporter.export(out_formats)

    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
