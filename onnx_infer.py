import json
import os
import pathlib
import warnings

import click
import librosa
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from tools.align_word import WordList, Word, Phoneme
from tools.config_utils import check_configs
from tools.decoder import AlignmentDecoder, NonLexicalDecoder
from tools.export_tool import Exporter
from tools.g2p import DictionaryG2P, PhonemeG2P
from tools.post_processing import find_all_duplicate_phonemes, remove_outliers_per_position


class InferenceBase:
    def __init__(self):
        self.vocab = None
        self.mel_cfg = None
        self.nll_decoder = None
        self.fa_decoder = None
        self.dataset = []
        self.predictions = []

    def load_config(self):
        pass

    def load_model(self):
        pass

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        return NotImplementedError

    def init_decoder(self):
        self.nll_decoder = NonLexicalDecoder(vocab=self.vocab,
                                             class_names=['None', *self.vocab['non_lexical_phonemes']],
                                             sample_rate=self.mel_cfg["sample_rate"], hop_size=self.mel_cfg["hop_size"])
        self.fa_decoder = AlignmentDecoder(vocab=self.vocab, sample_rate=self.mel_cfg["sample_rate"],
                                           hop_size=self.mel_cfg["hop_size"])

    def get_dataset(self, wav_folder, language, g2p="dictionary", dictionary_path=None, in_format="lab"):
        if dictionary_path is None:
            dictionary_path = self.vocab["dictionaries"].get(language, "")
        language = language if self.vocab['language_prefix'] else None

        if g2p == "dictionary":
            assert os.path.exists(dictionary_path), f"{pathlib.Path(dictionary_path).absolute()} does not exist."
            g2p = DictionaryG2P(language, dictionary_path)
        elif g2p == "phoneme":
            g2p = PhonemeG2P(language)
        else:
            raise f"g2p - {g2p} is not supported, which should be 'dictionary' or 'phoneme'."

        wav_paths = pathlib.Path(wav_folder).rglob("*.wav")
        for wav_path in wav_paths:
            try:
                lab_path = wav_path.with_suffix("." + in_format)
                if lab_path.exists():
                    with open(lab_path, "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = g2p(lab_text)
                    self.dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
                else:
                    warnings.warn(f"{pathlib.Path(wav_path).absolute()} does not exist.")
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
        print(f"Loaded {len(self.dataset)} samples.")

    def infer(self, non_lexical_phonemes, pad_times=1, pad_length=5):
        non_lexical_phonemes = [ph.strip() for ph in non_lexical_phonemes.split(",") if ph.strip()]
        assert set(non_lexical_phonemes).issubset(set(self.vocab['non_lexical_phonemes'])), \
            f"The non_lexical_phonemes contain elements that are not included in the vocab."

        pad_lengths = [round(pad_length / pad_times * i, 1) for i in range(0, pad_times)] if pad_times > 1 else [0]

        for i in tqdm(range(len(self.dataset)), desc="Processing", unit="it"):
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language, non_lexical_phonemes = self.dataset[i]

            # Load and resample audio
            wav, sr = librosa.load(wav_path, sr=self.mel_cfg['sample_rate'], mono=True)
            wav_length = len(wav) / self.mel_cfg['sample_rate']

            words_list: list[WordList] = []
            for pl in pad_lengths:
                padded_samples = int(pl * self.mel_cfg['sample_rate'])
                padded_frames = int(padded_samples / self.mel_cfg['hop_size'])
                padded_wav = np.pad(wav, (padded_samples, 0), mode='constant', constant_values=0)

                words, non_lexical_words = self._infer(padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx,
                                                       wav_length, non_lexical_phonemes)
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
                    ph_end = \
                        remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].end for words in words_list]])[0]
                    ph_start = max(ph_start, phonemes_all[-1].end if len(phonemes_all) > 0 else 0)
                    ph_end = max(ph_start + 0.0001, ph_end)
                    phonemes.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                    phonemes_all.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                word = Word(phonemes[0].start, phonemes[-1].end, words_list[0][w_idx].text)
                for ph in phonemes:
                    word.append_phoneme(ph)
                result_word.append(word)
            result_word.add_SP(wav_length)
            self.predictions.append((wav_path, wav_length, result_word))

    def export(self, output_folder, output_format=None):
        if output_format is None:
            output_format = ['textgrid']
        Exporter(self.predictions, output_folder).export(output_format)
        print("Output files are saved to the same folder as the input wav files.")


class InferenceOnnx(InferenceBase):
    def __init__(self, onnx_folder):
        super().__init__()
        self.model = None
        self.onnx_folder = pathlib.Path(onnx_folder)

    def load_config(self):
        check_configs(self.onnx_folder, suffix='json')
        with open(self.onnx_folder / 'VERSION', 'r', encoding='utf-8') as f:
            assert int(f.readline().strip()) == 5, f"onnx model version must be 5."
        with open(self.onnx_folder / 'vocab.json', 'r', encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        with open(self.onnx_folder / 'config.json', 'r', encoding='utf-8') as f:
            config = json.loads(f.read())

        self.mel_cfg = config['mel_spec_config']

    def load_model(self):
        self.model = self.create_session(self.onnx_folder / 'model.onnx')

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        results = self.run_onnx(self.model, {'waveform': [padded_wav]})

        words, _ = self.fa_decoder.decode(
            ph_frame_logits=results['ph_frame_logits'][:, :, padded_frames + 1:],
            ph_edge_logits=results['ph_edge_logits'][:, padded_frames + 1:],
            wav_length=wav_length, ph_seq=ph_seq, word_seq=word_seq, ph_idx_to_word_idx=ph_idx_to_word_idx
        )

        non_lexical_words = self.nll_decoder.decode(cvnt_logits=results['cvnt_logits'][:, :, padded_frames + 1:],
                                                    wav_length=wav_length,
                                                    non_lexical_phonemes=non_lexical_phonemes)
        return words, non_lexical_words

    @staticmethod
    def run_onnx(session, input_dict):
        output_names = [output.name for output in session.get_outputs()]
        return dict(zip(output_names, session.run(output_names, input_dict)))

    @staticmethod
    def create_session(onnx_path):
        providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), options, providers=providers)


@click.command()
@click.option("--onnx_folder", "-of", required=True, type=pathlib.Path, default=None, help="Path to ONNX models")
@click.option("--wav_folder", "-f", default="segments", type=str, help="Input folder path")
@click.option("--g2p", "-g", default="Dictionary", type=str, help="G2P class name")
@click.option("--non_lexical_phonemes", "-np", default="AP", type=str, help="non speech phonemes, exp. AP,EP")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
@click.option("--pad_times", "-pt", type=int, default=1, help="The number of times to pad blank audio before reasoning")
@click.option("--pad_length", "-pl", type=int, default=5,
              help="The max length of blank audio on the pad before inference")
def infer(onnx_folder, wav_folder, g2p, non_lexical_phonemes, language, dictionary, pad_times, pad_length):
    if onnx_folder is not None:
        inference = InferenceOnnx(onnx_folder)
    else:
        inference = InferenceOnnx(onnx_folder)
    inference.get_dataset(wav_folder=wav_folder, language=language, g2p=g2p, dictionary_path=dictionary)
    inference.infer(non_lexical_phonemes=non_lexical_phonemes, pad_times=pad_times, pad_length=pad_length)
    inference.export(output_folder=onnx_folder, output_format=['textgrid'])


if __name__ == '__main__':
    infer()
