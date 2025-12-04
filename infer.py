import os
import pathlib

import click
import lightning as pl
import torch

import networks.g2p
from tools.config_utils import check_configs, load_yaml
from tools.encoder import UnitsEncoder
from tools.export_tool import Exporter
from train import LitForcedAlignmentTask


@click.command()
@click.option("--ckpt", "-c", default=None, required=True, type=str, help="path to the checkpoint")
@click.option("--encoder", "-e", default=None, type=str, help="path to the encoder model")
@click.option("--folder", "-f", default="segments", type=str, help="path to the input folder")
@click.option("--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class")
@click.option("--non_lexical_phonemes", "-np", default="AP", type=str, help="non speech phonemes, exp. AP,EP")
@click.option("--language", "-l", default="zh", type=str, help="language of dictionary.(exp. zh ja en yue)")
@click.option("--dictionary", "-d", default=None, type=str,
              help="(only used when --g2p=='Dictionary') path to the dictionary")
@click.option("--pad_times", "-pt", type=int, default=1, help="The number of times to pad blank audio before reasoning")
@click.option("--pad_length", "-pl", type=int, default=5,
              help="The max length of blank audio on the pad before inference")
def main(ckpt, encoder, folder, g2p, non_speech_phonemes, language, dictionary):
    model_dir = pathlib.Path(ckpt).parent
    check_configs(model_dir)

    vocab = load_yaml(model_dir / "vocab.yaml")
    non_speech_phonemes = [ph.strip() for ph in non_speech_phonemes.split(",") if ph.strip()]

    if "Dictionary" in g2p:
        if dictionary is None:
            dictionary = model_dir / vocab["dictionaries"].get(language, "")
        assert os.path.exists(dictionary), f"{pathlib.Path(dictionary).absolute()} does not exist"

    assert set(non_speech_phonemes).issubset(set(vocab['non_lexical_phonemes'])), \
        f"The non_lexical_phonemes contain elements that are not included in the vocab."

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(
        **{"language": language, "dictionary": dictionary, "non_lexical_phonemes": non_speech_phonemes})
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    if encoder is not None:
        model.unitsEncoder = UnitsEncoder(model.hubert_config, model.mel_spec_config, encoder, model.device)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)
    Exporter(predictions).export(['textgrid'])
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
