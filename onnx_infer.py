import os
import pathlib

import click
import librosa
import onnxruntime as ort
import yaml
from tqdm import tqdm

import networks.g2p
from tools.alignment_decoder import AlignmentDecoder
from tools.config_utils import check_configs
from tools.export_tool import Exporter
from tools.post_processing import post_processing


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_onnx(session, input_dict):
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, session.run(output_names, input_dict)))


def create_session(onnx_path):
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), options, providers=providers)


@click.command()
@click.option("--onnx_folder", "-of", required=True, type=pathlib.Path, help="Path to ONNX models")
@click.option("--folder", "-f", default="segments", type=str, help="Input folder path")
@click.option("--g2p", "-g", default="Dictionary", type=str, help="G2P class name")
@click.option("--save_confidence", "-sc", is_flag=True, help="Save confidence.csv")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
def infer(onnx_folder, folder, g2p, save_confidence, language, dictionary):
    onnx_folder = pathlib.Path(onnx_folder)
    check_configs(onnx_folder)

    if "Dictionary" in g2p:
        if dictionary is None:
            vocab = load_yaml(onnx_folder / "vocab.yaml")
            dictionary = onnx_folder / vocab["dictionaries"].get(language, "")
        assert os.path.exists(dictionary), f"{pathlib.Path(dictionary).absolute()} does not exist"

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**{"language": language, "dictionary": dictionary})
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    config = load_yaml(onnx_folder / 'config.yaml')
    vocab = load_yaml(onnx_folder / 'vocab.yaml')
    language_prefix = vocab.get("language_prefix", True)

    hubert_cfg = config['hubert_config']
    mel_cfg = config['melspec_config']

    # Create ONNX sessions
    encoder = create_session(onnx_folder / f"{hubert_cfg['encoder']}-{hubert_cfg['channel']}.onnx")
    predictor = create_session(onnx_folder / 'model.onnx')

    # Process dataset
    decoder = AlignmentDecoder(vocab, mel_cfg)
    predictions = []
    ignored_phonemes = vocab['silent_phonemes'] + vocab['global_phonemes']

    for i in tqdm(range(len(dataset)), desc="Processing", unit="it"):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language = dataset[i]
        ph_seq = [f"{language}/{ph}" if ph not in ignored_phonemes and language_prefix else ph for ph in ph_seq]

        # Load and resample audio
        wav, sr = librosa.load(wav_path, sr=mel_cfg['sample_rate'], mono=True)
        wav_length = len(wav) / mel_cfg['sample_rate']

        # Run models
        feature = run_onnx(encoder, {'waveform': [wav]})["input_feature"]
        results = run_onnx(predictor, {'input_feature': feature})

        (
            ph_seq, ph_intervals, word_seq, word_intervals, confidence
        ) = decoder.decode(
            results['ph_frame_logits'], results['ph_edge_logits'], results['ctc_logits'],
            wav_length, ph_seq, word_seq, ph_idx_to_word_idx
        )

        ph_seq = [x.split("/")[-1] for x in ph_seq]

        predictions.append((wav_path,
                            wav_length,
                            confidence,
                            ph_seq,
                            ph_intervals,
                            word_seq,
                            word_intervals,))

    predictions, log = post_processing(predictions)
    if log:
        print("error:", "\n".join(log))

    Exporter(predictions).export(['textgrid', 'confidence'] if save_confidence else ['textgrid'])
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == '__main__':
    infer()
