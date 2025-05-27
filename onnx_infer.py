import os
import pathlib

import click
import onnxruntime as ort
import torch
import torchaudio
import yaml
from tqdm import tqdm

import networks.g2p
from tools.alignment_decoder import AlignmentDecoder
from tools.config_utils import check_configs, load_yaml
from tools.export_tool import Exporter
from tools.post_processing import post_processing


def load_config_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def encode(session, waveform):
    output_names = [output.name for output in session.get_outputs()]

    try:
        results = session.run(output_names, {'waveform': waveform})
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        raise

    output_dict = {name: result for name, result in zip(output_names, results)}

    return output_dict


def predict(session, input_feature):
    output_names = [output.name for output in session.get_outputs()]

    try:
        results = session.run(output_names, {'input_feature': input_feature})
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        raise

    output_dict = {name: result for name, result in zip(output_names, results)}

    return output_dict


def create_session(onnx_model_path):
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
    except Exception as e:
        print(f"An error occurred while creating ONNX Runtime session: {e}")
        raise

    return session


@click.command()
@click.option(
    "--onnx_folder",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the onnx",
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
    default=None,
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def infer(onnx_folder,
          folder,
          g2p,
          save_confidence,
          **kwargs,
          ):
    onnx_folder = pathlib.Path(onnx_folder)
    check_configs(onnx_folder)

    if "Dictionary" in g2p:
        if kwargs["dictionary"] is None:
            vocab = load_yaml(onnx_folder / "vocab.yaml")
            dictionary_path = onnx_folder / vocab["dictionaries"].get(kwargs["language"], "")
            kwargs["dictionary"] = dictionary_path
        assert os.path.exists(kwargs["dictionary"]), f"{pathlib.Path(kwargs['dictionary']).absolute()} does not exist"

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = load_config_from_yaml(onnx_folder / 'config.yaml')
    vocab = load_config_from_yaml(onnx_folder / 'vocab.yaml')

    hubert_config = config['hubert_config']
    melspec_config = config['melspec_config']
    dictionaries = vocab['dictionaries']
    language_prefix = vocab.get("language_prefix", True)

    encoder_name = hubert_config['encoder']

    dictionary = kwargs.get("dictionary", None)
    if dictionary is None:
        dictionary = dictionaries[kwargs['language']]
    kwargs['dictionary'] = dictionary

    encoder_session = create_session(onnx_folder / f"{encoder_name}-{hubert_config['channel']}.onnx")
    predict_session = create_session(onnx_folder / 'model.onnx')

    decoder = AlignmentDecoder(vocab, melspec_config)

    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))
    predictions = []
    ignored_phonemes = vocab['silent_phonemes'] + vocab['global_phonemes']

    for i in tqdm(range(len(dataset)), desc="Processing", unit="it"):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language = dataset[i]
        ph_seq = [f"{language}/{ph}" if ph not in ignored_phonemes and language_prefix else ph for ph in ph_seq]

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform[0][None, :][0]
        if sr != melspec_config['sample_rate']:
            waveform = torchaudio.transforms.Resample(sr, melspec_config['sample_rate'])(waveform)

        wav_length = waveform.shape[0] / melspec_config["sample_rate"]

        input_feature = encode(encoder_session, [waveform.cpu().numpy()])[
            "input_feature"]  # [B, T, C]
        results = predict(predict_session, input_feature)

        ph_frame_logits = torch.as_tensor(results['ph_frame_logits'], device=device)
        ph_edge_logits = torch.as_tensor(results['ph_edge_logits'], device=device)
        ctc_logits = torch.as_tensor(results['ctc_logits'], device=device)

        (
            ph_seq, ph_intervals, word_seq, word_intervals, confidence
        ) = decoder.decode(
            ph_frame_logits, ph_edge_logits, ctc_logits, wav_length, ph_seq, word_seq, ph_idx_to_word_idx
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

    exporter = Exporter(predictions)
    exporter.export(['textgrid'] if not save_confidence else ['textgrid', 'confidence'])

    print("Output files are saved to the same folder as the input wav files.")


if __name__ == '__main__':
    infer()
