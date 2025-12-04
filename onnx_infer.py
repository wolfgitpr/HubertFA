import json
import os
import pathlib

import click
import librosa
import numpy as np
import onnxruntime as ort
import yaml
from tqdm import tqdm

import networks.g2p
from tools.align_word import WordList, Word, Phoneme
from tools.config_utils import check_configs
from tools.decoder import AlignmentDecoder, NonLexicalDecoder
from tools.export_tool import Exporter
from tools.post_processing import find_all_duplicate_phonemes, remove_outliers_per_position


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
@click.option("--non_lexical_phonemes", "-np", default="AP", type=str, help="non speech phonemes, exp. AP,EP")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
@click.option("--pad_times", "-pt", type=int, default=1, help="The number of times to pad blank audio before reasoning")
@click.option("--pad_length", "-pl", type=int, default=5,
              help="The max length of blank audio on the pad before inference")
def infer(onnx_folder, folder, g2p, non_lexical_phonemes, language, dictionary, pad_times, pad_length):
    onnx_folder = pathlib.Path(onnx_folder)
    check_configs(onnx_folder, suffix='json')
    with open(onnx_folder / 'VERSION', 'r', encoding='utf-8') as f:
        assert int(f.readline().strip()) == 5, f"onnx model version must be 5."

    with open(onnx_folder / 'vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.loads(f.read())
    non_lexical_phonemes = [ph.strip() for ph in non_lexical_phonemes.split(",") if ph.strip()]

    if "Dictionary" in g2p:
        if dictionary is None:
            dictionary = onnx_folder / vocab["dictionaries"].get(language, "")
        assert os.path.exists(dictionary), f"{pathlib.Path(dictionary).absolute()} does not exist"

    assert set(non_lexical_phonemes).issubset(set(vocab['non_lexical_phonemes'])), \
        f"The non_lexical_phonemes contain elements that are not included in the vocab."

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(
        **{"language": language, "dictionary": dictionary, "non_lexical_phonemes": non_lexical_phonemes})
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    with open(onnx_folder / 'config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    language_prefix = vocab['language_prefix']
    mel_cfg = config['melspec_config']

    # Create ONNX sessions
    model = create_session(onnx_folder / 'model.onnx')

    # Process dataset
    nll_decoder = NonLexicalDecoder(vocab=vocab, class_names=['None', *vocab['non_lexical_phonemes']],
                                    sample_rate=mel_cfg["sample_rate"], hop_size=mel_cfg["hop_size"])
    fa_decoder = AlignmentDecoder(vocab=vocab, sample_rate=mel_cfg["sample_rate"], hop_size=mel_cfg["hop_size"])
    predictions = []
    ignored_phonemes = vocab['silent_phonemes'] + vocab['non_lexical_phonemes']

    pad_lengths = [round(pad_length / pad_times * i, 1) for i in range(0, pad_times)] if pad_times > 1 else [0]

    for i in tqdm(range(len(dataset)), desc="Processing", unit="it"):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language, non_lexical_phonemes = dataset[i]
        ph_seq = [f"{language}/{ph}" if ph not in ignored_phonemes and language_prefix else ph for ph in ph_seq]

        # Load and resample audio
        wav, sr = librosa.load(wav_path, sr=mel_cfg['sample_rate'], mono=True)
        wav_length = len(wav) / mel_cfg['sample_rate']

        words_list: list[WordList] = []
        for pl in pad_lengths:
            padded_samples = int(pl * mel_cfg['sample_rate'])
            padded_frames = int(padded_samples / mel_cfg['hop_size'])
            padded_wav = np.pad(wav, (padded_samples, 0), mode='constant', constant_values=0)

            results = run_onnx(model, {'waveform': [padded_wav]})

            words, _ = fa_decoder.decode(
                ph_frame_logits=results['ph_frame_logits'][:, :, padded_frames + 1:],
                ph_edge_logits=results['ph_edge_logits'][:, padded_frames + 1:],
                wav_length=wav_length, ph_seq=ph_seq, word_seq=word_seq, ph_idx_to_word_idx=ph_idx_to_word_idx
            )

            non_lexical_words = nll_decoder.decode(cvnt_logits=results['cvnt_logits'][:, :, padded_frames + 1:],
                                                   wav_length=wav_length,
                                                   non_lexical_phonemes=non_lexical_phonemes)
            for _words in non_lexical_words:
                for word in _words:
                    if word.text in ['AP', 'EP']:
                        words.add_AP(word)
            words.clear_language_prefix()
            words_list.append(words)

        ph_list = [words.phonemes for words in words_list]
        words_list = [words_list[i] for i in find_all_duplicate_phonemes(ph_list)]

        phonemes_all = []
        result_word = WordList()
        for w_idx in range(len(words_list[0])):
            phonemes = []
            for ph_idx in range(len(words_list[0][w_idx].phonemes)):
                ph_start = \
                    remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].start for words in words_list]])[0]
                ph_end = remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].end for words in words_list]])[0]
                ph_start = max(ph_start, phonemes_all[-1].end if len(phonemes_all) > 0 else 0)
                ph_end = max(ph_start + 0.0001, ph_end)
                phonemes.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                phonemes_all.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
            word = Word(phonemes[0].start, phonemes[-1].end, words_list[0][w_idx].text)
            for ph in phonemes:
                word.append_phoneme(ph)
            result_word.append(word)
        result_word.add_SP(wav_length)
        predictions.append((wav_path, wav_length, result_word))

    Exporter(predictions).export(['textgrid'])
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == '__main__':
    infer()
