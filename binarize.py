import pathlib
import shutil

import click
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from tools.binarize_util import load_wav, get_curves
from tools.config_utils import load_yaml
from tools.dataset import IndexedDatasetBuilder
from tools.encoder import UnitsEncoder
from tools.get_melspec import MelSpecExtractor
from tools.multiprocess_utils import chunked_multiprocess_run


class BaseBinarizer(object):
    def __init__(self, binary_config_path):
        self.binary_config = load_yaml(binary_config_path)
        self.mel_spec_config = self.binary_config['mel_spec_config']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_mel_spec = MelSpecExtractor(**self.mel_spec_config, device=self.device)
        self.unitsEncoder = UnitsEncoder(self.binary_config['hubert_config'], self.binary_config['mel_spec_config'],
                                         device=self.device)

        self.datasets = self.load_datasets()
        self.binary_folder = pathlib.Path(self.binary_config['binary_folder'])
        self.binary_folder.mkdir(parents=True, exist_ok=True)

        self.multiprocess_works: int = self.binary_config['multiprocess_works']
        self.multiprocess_max_size: int = self.binary_config['multiprocess_max_size']
        self.multiprocess_start_size: int = self.binary_config['multiprocess_start_size']

        self.valid_sets = []
        self.valid_set_size = self.binary_config['valid_set_size']

        self.hop_size = self.mel_spec_config["hop_size"]
        self.window_size = self.mel_spec_config["window_size"]
        self.sample_rate = self.mel_spec_config["sample_rate"]
        self.frame_length = self.hop_size / self.sample_rate
        self.max_length = self.binary_config['max_length']

        self.aug_args: dict = self.binary_config['augmentation_args']
        self.aug_num: int = 1 + (
            (self.aug_args['random_pitch_shifting']['num'] + self.aug_args['blank_padding']['num'] if self.aug_args[
                'enabled'] else 0)
        )

        shutil.copy(binary_config_path, self.binary_folder / 'config.yaml')
        self.export_config(self.binary_folder / 'datasets.yaml',
                           {"aug_num": self.aug_num, "datasets": self.datasets})

    def load_datasets(self):
        datasets_config_paths = self.binary_config["datasets_config_paths"]
        assert isinstance(datasets_config_paths, list), f"{datasets_config_paths} is not a list."

        valid_keys = {'language', 'label_type', 'raw_data_dir', 'test_prefixes'}
        valid_datasets = []
        for dataset_config_path in datasets_config_paths:
            assert pathlib.Path(dataset_config_path).exists(), f"{dataset_config_path} does not exist."
            datasets = load_yaml(dataset_config_path)["datasets"]
            for item in datasets:
                assert isinstance(item, dict), f"{item} is not a dict."
                assert valid_keys - item.keys() == set(), f"datasets not contains keys - {valid_keys - item.keys()}: \n[{item}].\n"
                assert pathlib.Path(item['raw_data_dir']).exists(), f"{item['raw_data_dir']} does not exist."
                assert item['label_type'] in ['full', 'evaluate'], f"{item['label_type']} not in ['full','evaluate']."
                valid_datasets.append(item)

        assert len(valid_datasets) > 0, f"datasets are empty."
        return valid_datasets

    @staticmethod
    def export_config(file_path: str | pathlib.Path, config: dict):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    def binarize(
            self,
            prefix: str,
            meta_data: pd.DataFrame,
            binary_data_folder: str | pathlib.Path,
    ):
        print(f"Binarizing {prefix} set...")
        train = True if prefix == "train" else False

        args = []
        builder = IndexedDatasetBuilder(binary_data_folder, aug_num=self.aug_num, prefix=prefix)

        for _, item in meta_data.iterrows():
            args.append((item, train))

        try:
            if self.multiprocess_works > 0 and len(args) > self.multiprocess_start_size:
                # code for parallel processing
                for item in tqdm(
                        chunked_multiprocess_run(self.process_item, args, num_workers=self.multiprocess_works,
                                                 q_max_size=self.multiprocess_max_size),
                        total=len(args)
                ):
                    if item is not None:
                        builder.add_item(item)
            else:
                # code for single cpu processing
                for a in tqdm(args):
                    item = self.process_item(*a)
                    if item is not None:
                        builder.add_item(item)
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()

        total_time = sum(builder.aug_wav_lengths)
        print(
            f"Successfully binarized {prefix} set, "
            f"total time {total_time:.2f}s ({(total_time / 3600):.2f} h), saved to {builder.path}"
        )

    def process(self):
        # load metadata of each item
        meta_data_df: pd.DataFrame = self.get_meta_data()
        meta_data_df: pd.DataFrame = self.post_process_meta_data(meta_data_df)

        meta_data_evaluate = meta_data_df[meta_data_df["label_type"] == "evaluate"]
        meta_data_df = meta_data_df.drop(meta_data_evaluate.index)

        # split train and valid set
        valid_set_size = int(self.valid_set_size)
        if self.valid_set_size > 0:
            meta_data_valid = (
                meta_data_df[meta_data_df["label_type"] == "full"]
                .sample(frac=1)
                .iloc[:valid_set_size, :]
            )
        else:
            meta_data_valid = meta_data_df[meta_data_df["validation"] == True]

        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)
        meta_data_valid = meta_data_valid.reset_index(drop=True)
        meta_data_evaluate = meta_data_evaluate.reset_index(drop=True)

        assert len(meta_data_valid) > 0, "No valid data found."

        # binarize set
        self.binarize("evaluate", meta_data_evaluate, self.binary_folder)
        self.binarize("valid", meta_data_valid, self.binary_folder)
        self.binarize("train", meta_data_train, self.binary_folder)

    def process_item(self, **kwargs):
        pass

    def get_meta_data(self):
        print("Loading metadata...")
        meta_data_df = pd.DataFrame()
        for dataset in self.datasets:
            test_prefixes = dataset['test_prefixes']
            tuple_prefixes = tuple([x for x in test_prefixes if x] if test_prefixes is not None else [])

            raw_data_dir = pathlib.Path(dataset['raw_data_dir'])
            csv_path = raw_data_dir / "transcriptions.csv"
            wav_folder = raw_data_dir / "wavs"
            assert csv_path.exists() and wav_folder.exists(), f"{csv_path.absolute()} or {wav_folder.absolute()} does not exist."

            df = pd.read_csv(csv_path, dtype=str)
            original_count = len(df)

            df['name'] = df['name'].astype(str)
            invalid_names = df['name'].apply(lambda x: not isinstance(x, str) or x == 'nan' or x.strip() == '')

            if invalid_names.any():
                invalid_count = invalid_names.sum()
                print(f"Warning: Found {invalid_count} invalid name(s) in {csv_path}. Removing these rows.")
                print(f"Invalid names: {df[invalid_names]['name'].tolist()}")

                df = df[~invalid_names].copy()
                print(f"Removed {invalid_count} invalid rows. Remaining: {len(df)}/{original_count}")

            if len(df) == 0:
                print(f"Error: No valid data remaining in {csv_path} after cleaning!")
                continue

            assert "ph_seq" in df.columns, f"{csv_path.absolute()} does not contain 'ph_seq'."
            assert "ph_dur" in df.columns, f"full label csv: {csv_path.absolute()} does not contain 'ph_dur'."

            df['language'] = dataset['language']
            df["label_type"] = dataset['label_type']
            df["wav_path"] = df["name"].apply(lambda name: str(wav_folder / (str(name) + ".wav")))
            df["validation"] = df["name"].apply(lambda name: name.startswith(tuple_prefixes)) \
                if len(tuple_prefixes) > 0 else False
            meta_data_df = pd.concat([meta_data_df, df]) if len(meta_data_df) >= 1 else df

        meta_data_df["ph_seq"] = meta_data_df["ph_seq"].apply(
            lambda raw_str: ([ph for ph in raw_str.split(" ")] if isinstance(raw_str, str) else [])
        )
        meta_data_df["ph_dur"] = meta_data_df["ph_dur"].apply(
            lambda x: [float(i) for i in x.split(" ")] if isinstance(x, str) else []
        )
        meta_data_df.reset_index(drop=True, inplace=True)
        meta_data_df = meta_data_df.sort_values(by="label_type").reset_index(drop=True)

        print(f"Final metadata contains {len(meta_data_df)} valid rows.")
        return meta_data_df

    def post_process_meta_data(self, meta_data_df) -> pd.DataFrame:
        pass


class NonLexicalLabelBinarizer(BaseBinarizer):
    def __init__(self, binary_config):
        super().__init__(binary_config)
        self.non_lexical_phonemes = self.binary_config['non_lexical_phonemes']
        assert len(self.non_lexical_phonemes) > 1, "non_lexical_phonemes must have at least one phoneme."

        self.non_lexical_phonemes_dict = {**{"None": 0},
                                          **{ph: i + 1 for i, ph in enumerate(self.non_lexical_phonemes)}}

        self.vocab = self.get_vocab()
        self.export_config(self.binary_folder / 'vocab.yaml', self.vocab)

    def get_vocab(self):
        print("Generating vocab...")
        vocab = {"non_lexical_phonemes": self.non_lexical_phonemes,
                 "non_lexical_phonemes_dict": self.non_lexical_phonemes_dict}
        return vocab

    def make_non_lexical_ph_data(self, frames, ph_id_seq, ph_duration):
        if len(ph_id_seq) == 0:
            return np.zeros((len(self.vocab.keys()) + 1, frames), dtype="int32"), []

        ph_id_seq = np.array(ph_id_seq, dtype="int32")
        ph_dur = np.array(ph_duration, dtype="float32")

        ph_time = np.concatenate(([0], ph_dur)).cumsum()
        ph_frame = ph_time / self.frame_length

        ph_frame_int = np.round(ph_frame).astype("int32")
        ph_frame_int = np.clip(ph_frame_int, 0, frames)

        num_phones = len(self.vocab.keys()) + 1
        non_lexical_target = np.zeros((num_phones, frames), dtype="int32")
        non_lexical_intervals = []

        for i, ph_id in enumerate(ph_id_seq):
            start_frame = ph_frame_int[i]
            end_frame = ph_frame_int[i + 1]
            if start_frame < end_frame:
                non_lexical_target[ph_id, start_frame:end_frame] = 1
                if ph_id > 0:
                    non_lexical_intervals.append([start_frame, end_frame])

        return non_lexical_target, np.array(non_lexical_intervals)

    @torch.no_grad()
    def process_item(self, _item, train):
        try:
            if not pathlib.Path(wav_path := _item['wav_path']).exists():
                print(f"Skipping {wav_path}, because it doesn't exist")
                return None

            waveform, wav_length, n_frames = load_wav(wav_path, self.sample_rate, self.hop_size,
                                                      self.device)  # (L,) seconds
            if wav_length > self.max_length:
                print(f"Item {wav_path} has a length of {wav_length} s, which is too long, skip it.")
                return None

            if len(_item.ph_seq) == 0 or len(_item.ph_dur) != len(_item.ph_seq):
                return None

            non_lexical_target, non_lexical_intervals = self.make_non_lexical_ph_data(n_frames,
                                                                                      _item.non_lexical_phonemes_id_seq,
                                                                                      _item.ph_dur)  # [B, C, T]
            if non_lexical_target is None:
                print(f"Skipping {wav_path}, make non_lexical_ph data failed.")
                return None

            units = self.unitsEncoder.forward(waveform.unsqueeze(0), self.sample_rate, self.hop_size,
                                              aug=self.binary_config['augmentation_args']['enabled'] and train,
                                              aug_args=self.binary_config['augmentation_args'])  # [B, T, C]
            mel_spec = self.get_mel_spec(waveform).cpu().numpy() if not train else np.array([[[0]]])  # [B, C, T]
            B, T, C = units.shape
            assert B == (
                self.aug_num if train else 1), f"Batch of input_feature must be equal to aug_num - {self.aug_num}."
            assert T > 0 and T == n_frames, f"Length of unit {T} must be greater than 0."

            return {
                'name': [str(_item["name"])] * B,
                'input_feature': units.transpose(1, 2).cpu().numpy().astype("float16"),  # [B, C, T]
                'mel_spec': np.repeat(mel_spec, B, axis=0).astype("float32"),  # [B, C, T]
                "non_lexical_target": np.repeat(non_lexical_target[np.newaxis, :], B, axis=0).astype("int32"),
                "non_lexical_intervals": np.repeat(non_lexical_intervals[np.newaxis, :], B, axis=0).astype("int32"),
                "wav_length": wav_length
            }

        except Exception as e:
            print(f"error: {_item.get('name', 'unknown')}: {str(e)}")
            return None

    def post_process_meta_data(self, meta_data_df) -> pd.DataFrame:
        meta_data_df["non_lexical_phonemes_id_seq"] = meta_data_df["ph_seq"].apply(
            lambda ph_seq: ([self.vocab['non_lexical_phonemes_dict'].get(ph, 0) for ph in ph_seq]))
        return meta_data_df


class ForcedAlignmentBinarizer(BaseBinarizer):
    def __init__(self, binary_config):
        super().__init__(binary_config)
        self.extra_phonemes = self.binary_config['extra_phonemes']
        self.silent_phonemes = self.binary_config['silent_phonemes']

        self.language_prefix = self.binary_config['language_prefix']
        self.dictionaries = self.binary_config['dictionaries']
        self.merged_phoneme_groups = self.binary_config['merged_phoneme_groups'] if self.binary_config[
            'merged_phoneme'] else []

        self.hubert_channel = self.binary_config['hubert_config']["channel"]

        self.vocab = self.get_vocab()
        self.export_config(self.binary_folder / 'vocab.yaml', self.vocab)

        for dict_path in self.dictionaries.values():
            shutil.copy(dict_path, self.binary_folder)

    def get_vocab(self):
        print("Generating vocab...")
        dataset_phonemes = []
        dict_phonemes = []

        for ph in self.extra_phonemes:
            if '/' in ph:
                lang, name = ph.split('/', maxsplit=1)
                if lang not in self.dictionaries:
                    raise ValueError(
                        f"Invalid phoneme tag '{ph}' in extra phonemes: "
                        f"unrecognized language name '{lang}'."
                    )
                if name in dataset_phonemes:
                    raise ValueError(
                        f"Invalid phoneme tag '{ph}' in extra phonemes: "
                        f"short name conflicts with existing tag."
                    )
            dataset_phonemes.append(ph)
            dict_phonemes.append(ph)

        for dataset in self.datasets:
            language = dataset.get("language", "blank")
            raw_data_dir = dataset["raw_data_dir"]

            csv_path = pathlib.Path(raw_data_dir) / "transcriptions.csv"
            assert csv_path.exists(), f"{csv_path.absolute()} does not exist."

            df = pd.read_csv(csv_path)
            ph_seq = list(set(" ".join(df["ph_seq"]).split(" ")))

            dataset_phonemes.extend(
                [ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                 else f"{language}/{ph}" for ph in ph_seq]
            )

        for lang, dict_path in self.dictionaries.items():
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    _word, _phonemes = line.strip("\n").split("\t")
                    _phonemes = _phonemes.split()
                    if '/' in _phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{_phonemes}' in dictionary '{dict_path}': "
                            f"should not contain the reserved character '/'."
                        )
                    dict_phonemes.extend(
                        [ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                         else f"{lang}/{ph}" for ph in _phonemes]
                    )

        dataset_phonemes = set(dataset_phonemes)
        for p in self.silent_phonemes:
            if p in dataset_phonemes:
                dataset_phonemes.remove(p)
        dataset_phonemes = sorted(dataset_phonemes)
        dataset_phonemes = ["SP", *dataset_phonemes]

        self.merged_phoneme_groups.insert(0, list({"SP", *self.silent_phonemes}))

        vocab = dict(zip(dataset_phonemes, range(len(dataset_phonemes))))  # phoneme: phoneme_id

        for i, merged_phoneme_group in enumerate(self.merged_phoneme_groups):
            vocab.update({ph: i for ph in merged_phoneme_group})

        for ph in dataset_phonemes:
            if ph not in vocab:
                vocab[ph] = len(vocab)

        vocab_dict = {"vocab": vocab,
                      "vocab_size": len(dataset_phonemes),
                      "language_prefix": self.language_prefix,
                      "silent_phonemes": list({"SP", *self.silent_phonemes}),
                      "merged_phoneme_groups": self.merged_phoneme_groups,
                      "dictionaries": {k: pathlib.Path(v).name for k, v in self.dictionaries.items()},
                      }

        print(f"vocab_size is {len(dataset_phonemes)}:")
        print(f"+ {[x for x in dataset_phonemes if x not in dict_phonemes and x not in self.silent_phonemes]}")
        print(f"- {[x for x in dict_phonemes if x not in dataset_phonemes and x not in self.silent_phonemes]}")
        return vocab_dict

    def make_ph_data(self, vocab, frames, raw_ph_id_seq, raw_ph_dur):
        # ph_seq: [S]
        ph_id_seq = np.array(raw_ph_id_seq).astype("int32")
        not_sp_idx = ph_id_seq != 0
        ph_id_seq = ph_id_seq[not_sp_idx]

        # ph_edge: [T]
        ph_dur = np.array(raw_ph_dur).astype("float32")
        ph_time = np.array(np.concatenate(([0], ph_dur))).cumsum()
        ph_frame = ph_time / self.frame_length
        ph_interval = np.stack((ph_frame[:-1], ph_frame[1:]))
        ph_time = ph_time[:-1]
        ph_time = ph_time[not_sp_idx]

        ph_interval = ph_interval[:, not_sp_idx]
        ph_id_seq = ph_id_seq
        ph_frame = np.unique(ph_interval.flatten())
        if ph_frame[-1] >= frames:
            ph_frame = ph_frame[:-1]

        if len(ph_id_seq) <= 0:
            return None, None, None, None, None

        ph_edge = np.zeros([frames], dtype="float32")
        if len(ph_id_seq) > 0:
            if ph_frame[-1] + 0.5 > frames:
                ph_frame = ph_frame[:-1]
            if ph_frame[0] - 0.5 < 0:
                ph_frame = ph_frame[1:]
            ph_time_int = np.round(ph_frame).astype("int32")
            ph_time_fractional = ph_frame - ph_time_int

            ph_edge[ph_time_int] = 0.5 + ph_time_fractional
            ph_edge[ph_time_int - 1] = 0.5 - ph_time_fractional
            ph_edge = ph_edge * 0.8 + 0.1

        # ph_frame: [T]
        ph_frame = np.zeros(frames, dtype="int32")
        if len(ph_id_seq) > 0:
            for ph_id, st, ed in zip(
                    ph_id_seq, ph_interval[0], ph_interval[1]
            ):
                if st < 0:
                    st = 0
                if ed > frames:
                    ed = frames
                ph_frame[int(np.round(st)): int(np.round(ed))] = ph_id

        # ph_mask: [vocab_size]
        ph_mask = np.zeros(vocab["vocab_size"], dtype="int32")
        if len(ph_id_seq) > 0:
            ph_mask[ph_id_seq] = 1
        ph_mask[0] = 1

        return ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time

    @torch.no_grad()
    def process_item(self, _item, train):
        try:
            if not pathlib.Path(wav_path := _item["wav_path"]).exists():
                print(f"Skipping {wav_path}, because it doesn't exist")
                return None

            waveform, wav_length, n_frames = load_wav(wav_path, self.sample_rate, self.hop_size,
                                                      self.device)  # (L,) seconds
            if wav_length > self.max_length:
                print(f"Item {wav_path} has a length of {wav_length}s, which is too long, skip it.")
                return None

            curves = get_curves(waveform, n_frames, self.window_size, self.hop_size)  # [B, C, T]

            if len(_item.ph_id_seq) == 0 or len(_item.ph_dur) != len(_item.ph_id_seq):
                return None
            ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time = self.make_ph_data(
                self.vocab, n_frames, _item.ph_id_seq, _item.ph_dur
            )
            if ph_id_seq is None:
                print(f"Skipping {wav_path}, make ph data failed.")
                return None

            units = self.unitsEncoder.forward(waveform.unsqueeze(0), self.sample_rate, self.hop_size,
                                              aug=self.binary_config['augmentation_args']['enabled'] and train,
                                              aug_args=self.binary_config['augmentation_args'])  # [B, T, C]
            mel_spec = self.get_mel_spec(waveform).cpu().numpy() if not train else np.array([[[0]]])  # [B, C, T]

            B, T, C = units.shape
            assert B == (
                self.aug_num if train else 1), f"Batch of input_feature must be equal to aug_num - {self.aug_num}."
            assert T > 0 and T == n_frames, f"Length of unit {T} must be greater than 0."
            assert C == self.hubert_channel, f"Item {wav_path} has unexpected channel of {C}, which should be {self.hubert_channel}."

            return {
                'name': [str(_item["name"])] * B,
                'input_feature': units.transpose(1, 2).cpu().numpy().astype("float16"),  # [B, C, T]
                'curves': np.repeat(curves.cpu().numpy(), B, axis=0).astype("float16"),  # [B, 1, T]
                'mel_spec': np.repeat(mel_spec, B, axis=0).astype("float32"),  # [B, C, T]
                'ph_id_seq': np.repeat([ph_id_seq], B, axis=0).astype("int32"),  # [B, N]
                'ph_edge': np.repeat([ph_edge], B, axis=0).astype("float32"),  # [B, T]
                'ph_frame': np.repeat([ph_frame], B, axis=0).astype("int32"),  # [B, T]
                'ph_mask': np.repeat([ph_mask], B, axis=0).astype("int32"),  # [B, T]
                'ph_time': np.repeat([ph_time], B, axis=0).astype("float32"),  # [B, N]
                'ph_time_raw': np.concatenate(([0], _item.ph_dur)).cumsum()[:-1].astype("float32"),  # [B, N]
                'ph_seq_raw': _item.ph_seq,
                'ph_seq': [[ph for ph in _item.ph_seq if self.vocab["vocab"][ph] != 0]] * B,
                "wav_length": np.repeat([wav_length], B, axis=0).astype("float32")
            }

        except Exception as e:
            print(f"error: {_item.get('name', 'unknown')}: {str(e)}")
            return None

    def post_process_meta_data(self, meta_data_df) -> pd.DataFrame:
        meta_data_df["ph_seq"] = meta_data_df.apply(
            lambda row: [
                ph if ph in self.silent_phonemes or "/" in ph or not self.language_prefix
                else f"{row['language']}/{ph}"
                for ph in row["ph_seq"]
            ],
            axis=1
        )

        meta_data_df["ph_id_seq"] = meta_data_df["ph_seq"].apply(
            lambda ph_seq: [self.vocab['vocab'][ph] for ph in ph_seq]
        )
        return meta_data_df


@click.command()
@click.option("--config", "-c", type=str, required=True, help="binarize config path")
@click.option("--model", "-m", type=str, required=True,
              help="model type: nll[non_lexical_labeler model, first step] fa[forced_alignment model, second step]")
def binarize(config: str, model: str):
    assert model in ['nll', 'fa'], "model type must in ['nll', 'fa'], please read help info or README.md."

    if model == "nll":
        NonLexicalLabelBinarizer(config).process()
    elif model == "fa":
        ForcedAlignmentBinarizer(config).process()
    else:
        raise Exception(f"unknown model: {model}")


if __name__ == "__main__":
    binarize()
