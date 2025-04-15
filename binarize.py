import os
import pathlib

import click
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from tools.dataset import IndexedDatasetBuilder
from tools.encoder import UnitsEncoder
from tools.get_melspec import MelSpecExtractor
from tools.load_wav import load_wav
from tools.multiprocess_utils import chunked_multiprocess_run

unitsEncoder = None
get_melspec = None


class ForcedAlignmentBinarizer:
    def __init__(self, binary_config):
        self.vocab = None

        self.binary_config = binary_config

        self.multiprocess_works = binary_config.get("multiprocess_works", 0)
        self.multiprocess_max_size = binary_config.get("multiprocess_max_size", 100)
        self.multiprocess_start_size = binary_config.get("multiprocess_start_size", 100)

        self.datasets = binary_config['datasets']
        self.binary_folder = pathlib.Path(binary_config['binary_folder'])

        self.valid_sets = []
        self.valid_set_size = binary_config['valid_set_size']

        self.extra_phonemes = binary_config['extra_phonemes']
        self.ignored_phonemes = binary_config['ignored_phonemes']
        self.melspec_config = binary_config['melspec_config']

        self.dictionaries = binary_config['dictionaries']
        self.merged_phoneme_groups = binary_config['merged_phoneme_groups'] if binary_config['merged_phoneme'] else []

        self.max_length = binary_config['max_length']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate = self.melspec_config["sample_rate"]
        self.frame_length = self.melspec_config["hop_length"] / self.sample_rate

        self.hop_size = binary_config['melspec_config']["hop_length"]

        self.hubert_channel = binary_config['hubert_config']["channel"]

    def get_vocab(self):
        print("Generating vocab...")
        dataset_phonemes = []
        dict_phonemes = []

        if self.extra_phonemes:
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

            dataset_phonemes.extend([f"{language}/{ph}" if ph not in self.ignored_phonemes else ph for ph in ph_seq])

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
                        [f"{lang}/{ph}" if ph not in self.ignored_phonemes else ph for ph in _phonemes])

        dataset_phonemes = set(dataset_phonemes)
        for p in self.ignored_phonemes:
            if p in dataset_phonemes:
                dataset_phonemes.remove(p)
        dataset_phonemes = sorted(dataset_phonemes)
        dataset_phonemes = ["SP", *dataset_phonemes]

        self.merged_phoneme_groups.insert(0, ["SP", *self.ignored_phonemes])

        vocab = dict(zip(dataset_phonemes, range(len(dataset_phonemes))))  # phoneme: phoneme_id

        for i, merged_phoneme_group in enumerate(self.merged_phoneme_groups):
            vocab.update({ph: i for ph in merged_phoneme_group})

        for ph in dataset_phonemes:
            if ph not in vocab:
                vocab[ph] = len(vocab)

        vocab_dict = {"vocab": vocab,
                      "vocab_size": len(dataset_phonemes),
                      "ignored_phonemes": ["SP", *self.ignored_phonemes],
                      "merged_phoneme_groups": self.merged_phoneme_groups,
                      }

        print(f"vocab_size is {len(dataset_phonemes)}:")
        print(f"+ {[x for x in dataset_phonemes if x not in dict_phonemes and x not in self.ignored_phonemes]}")
        print(f"- {[x for x in dict_phonemes if x not in dataset_phonemes and x not in self.ignored_phonemes]}")

        return vocab_dict

    def process(self):
        self.vocab = self.get_vocab()
        with open(self.binary_folder / "vocab.yaml", "w", encoding="utf-8") as file:
            yaml.dump(self.vocab, file)

        # load metadata of each item
        meta_data_df = self.get_meta_data()

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

        # binarize valid set
        self.binarize("evaluate", meta_data_evaluate, self.binary_folder)

        # binarize valid set
        self.binarize("valid", meta_data_valid, self.binary_folder)

        # binarize train set
        self.binarize("train", meta_data_train, self.binary_folder)

    def make_ph_data(self, vocab, T, label_type_id, raw_ph_id_seq, raw_ph_dur):
        if label_type_id == 0:
            # ph_seq: [S]
            ph_id_seq = np.array([]).astype("int32")

            # ph_edge: [T]
            ph_edge = np.zeros([T], dtype="float32")

            # ph_frame: [T]
            ph_frame = np.zeros(T, dtype="int32")

            # ph_time: [T]
            ph_time = np.zeros(T, dtype="float32")

            # ph_mask: [vocab_size]
            ph_mask = np.ones(vocab["vocab_size"], dtype="int32")
        elif label_type_id == 1:
            # ph_seq: [S]
            ph_id_seq = np.array(raw_ph_id_seq).astype("int32")
            ph_id_seq = ph_id_seq[ph_id_seq != 0]

            if len(ph_id_seq) <= 0:
                return None, None, None, None, None

            # ph_edge: [T]
            ph_edge = np.zeros([T], dtype="float32")

            # ph_frame: [T]
            ph_frame = np.zeros(T, dtype="int32")

            # ph_time: [T]
            ph_time = np.zeros(T, dtype="float32")

            # ph_mask: [vocab_size]
            ph_mask = np.zeros(vocab["vocab_size"], dtype="int32")
            ph_mask[ph_id_seq] = 1
            ph_mask[0] = 1
        elif label_type_id >= 2:
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
            if ph_frame[-1] >= T:
                ph_frame = ph_frame[:-1]

            if len(ph_id_seq) <= 0:
                return None, None, None, None, None

            ph_edge = np.zeros([T], dtype="float32")
            if len(ph_id_seq) > 0:
                if ph_frame[-1] + 0.5 > T:
                    ph_frame = ph_frame[:-1]
                if ph_frame[0] - 0.5 < 0:
                    ph_frame = ph_frame[1:]
                ph_time_int = np.round(ph_frame).astype("int32")
                ph_time_fractional = ph_frame - ph_time_int

                ph_edge[ph_time_int] = 0.5 + ph_time_fractional
                ph_edge[ph_time_int - 1] = 0.5 - ph_time_fractional
                ph_edge = ph_edge * 0.8 + 0.1

            # ph_frame: [T]
            ph_frame = np.zeros(T, dtype="int32")
            if len(ph_id_seq) > 0:
                for ph_id, st, ed in zip(
                        ph_id_seq, ph_interval[0], ph_interval[1]
                ):
                    if st < 0:
                        st = 0
                    if ed > T:
                        ed = T
                    ph_frame[int(np.round(st)): int(np.round(ed))] = ph_id

            # ph_mask: [vocab_size]
            ph_mask = np.zeros(vocab["vocab_size"], dtype="int32")
            if len(ph_id_seq) > 0:
                ph_mask[ph_id_seq] = 1
            ph_mask[0] = 1
        else:
            return None, None, None, None, None
        return ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time

    def binarize(
            self,
            prefix: str,
            meta_data: pd.DataFrame,
            binary_data_folder: str | pathlib.Path,
    ):
        print(f"Binarizing {prefix} set...")

        args = []
        builder = IndexedDatasetBuilder(binary_data_folder, prefix=prefix)

        for _, item in meta_data.iterrows():
            args.append(item)

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
                    item = self.process_item(a)
                    if item is not None:
                        builder.add_item(item)
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()

        total_time = sum(builder.wav_lengths)
        print(
            f"Successfully binarized {prefix} set, "
            f"total time {total_time:.2f}s ({(total_time / 3600):.2f}h), saved to {builder.path}"
        )

    @torch.no_grad()
    def process_item(self, _item):
        global unitsEncoder
        if unitsEncoder is None:
            unitsEncoder = UnitsEncoder(
                self.binary_config['hubert_config']["encoder"],
                self.binary_config['hubert_config']["model_path"],
                self.binary_config['hubert_config']["sample_rate"],
                self.binary_config['hubert_config']["hop_size"],
                self.device)

        global get_melspec
        if get_melspec is None:
            get_melspec = MelSpecExtractor(**self.binary_config['melspec_config'], device=self.device)

        try:
            if not os.path.exists(wav_path := _item["wav_path"]):
                return None

            waveform = load_wav(wav_path, self.device, self.sample_rate)  # (L,)

            wav_length = len(waveform) / self.sample_rate  # seconds
            if wav_length > self.max_length:
                print(
                    f"Item {wav_path} has a length of {wav_length}s, which is too long, skip it."
                )
                return None

            # units encode
            units = unitsEncoder.encode(waveform.unsqueeze(0), self.sample_rate, self.hop_size)  # [B, C, T]
            melspec = get_melspec(waveform)  # [B, C, T]

            B, C, T = units.shape
            assert C == self.hubert_channel, f"Item {wav_path} has unexpect channel of {C}, which should be {self.hubert_channel}."

            label_type_id = {"blank": 0, "weak": 1, "full": 2, "evaluate": 3}[_item.label_type]
            if label_type_id >= 2:
                if len(_item.ph_dur) != len(_item.ph_id_seq): label_type_id = 1
                if not _item.ph_id_seq: label_type_id = 0

            ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time = self.make_ph_data(
                self.vocab, units.shape[-1], label_type_id, _item.ph_id_seq, _item.ph_dur
            )
            if ph_id_seq is None: return None

            return {
                'name': str(_item["name"]),
                'input_feature': units.cpu().numpy().astype("float32"),
                'melspec': melspec.cpu().numpy().astype("float32"),
                'ph_id_seq': ph_id_seq.astype("int32"),
                'ph_edge': ph_edge.astype("float32"),
                'ph_frame': ph_frame.astype("int32"),
                'ph_mask': ph_mask.astype("int32"),
                'ph_time': ph_time.astype("float32"),
                'ph_time_raw': np.concatenate(([0], _item.ph_dur)).cumsum()[:-1].astype("float32"),
                'ph_seq_raw': _item.ph_seq,
                'ph_seq': [ph for ph in _item.ph_seq if self.vocab["vocab"][ph] != 0],
                "label_type": label_type_id,
                "wav_length": wav_length
            }

        except Exception as e:
            print(f"error: {_item.get('name', 'unknown')}: {str(e)}")
            return None

    def get_meta_data(self):
        print("Loading metadata...")
        meta_data_df = pd.DataFrame()
        for dataset in self.datasets:
            language = dataset.get("language", "blank")
            label_type = dataset["label_type"]
            raw_data_dir = pathlib.Path(dataset["raw_data_dir"])
            test_prefixes = dataset.get("test_prefixes", [])

            assert raw_data_dir.exists(), f"{raw_data_dir} does not exist."
            assert label_type in ["full", "weak", "evaluate", "blank"], \
                f"{label_type} not in ['full', 'weak', 'evaluate','blank]."
            if label_type == "blank":
                df = pd.DataFrame(
                    columns=["name", "ph_seq", "ph_id_seq", "label_type", "wav_length", "validation"])
                wavs_path = [i for i in raw_data_dir.rglob("*.wav")]
                df["name"] = [os.path.splitext(os.path.basename(i))[0] for i in wavs_path]
                df["wav_length"] = 0
                df["validation"] = False
            else:
                tuple_prefixes = tuple([x for x in test_prefixes if x] if test_prefixes is not None else [])

                csv_path = raw_data_dir / "transcriptions.csv"
                wav_folder = raw_data_dir / "wavs"
                assert csv_path.exists() and wav_folder.exists(), f"{csv_path.absolute()} or {wav_folder.absolute()} does not exist."

                df = pd.read_csv(csv_path, dtype=str)
                assert "ph_seq" in df.columns, f"{csv_path.absolute()} does not contain 'ph_seq'."
                if label_type == "full":
                    assert "ph_dur" in df.columns, f"full label csv: {csv_path.absolute()} does not contain 'ph_dur'."

                if len(tuple_prefixes) > 0:
                    df["validation"] = df["name"].apply(lambda name: name.startswith(tuple_prefixes))
                else:
                    df["validation"] = False

            df["label_type"] = label_type
            df["wav_path"] = df["name"].apply(lambda name: str(wav_folder / (str(name) + ".wav")))

            df["ph_seq"] = df["ph_seq"].apply(
                lambda raw_str: ([ph for ph in raw_str.split(" ")] if isinstance(raw_str, str) else [])
            )

            df["ph_seq"] = df["ph_seq"].apply(
                lambda ph_seq: ([f"{language}/{ph}" if ph not in self.ignored_phonemes else ph for ph in ph_seq])
            )

            df["ph_id_seq"] = df["ph_seq"].apply(lambda ph_seq: ([self.vocab['vocab'][ph] for ph in ph_seq]))

            meta_data_df = pd.concat([meta_data_df, df]) if len(meta_data_df) >= 1 else df

        meta_data_df.reset_index(drop=True, inplace=True)

        if "ph_dur" in meta_data_df.columns:
            meta_data_df["ph_dur"] = meta_data_df["ph_dur"].apply(
                lambda x: (
                    [float(i) for i in x.split(" ")] if isinstance(x, str) else []
                )
            )
        meta_data_df = meta_data_df.sort_values(by="label_type").reset_index(drop=True)

        return meta_data_df


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="configs/binarize_config.yaml",
    show_default=True,
    help="binarize config path",
)
def binarize(config: str):
    config = load_yaml(config)

    datasets_config = pathlib.Path(config["datasets_config"])
    assert datasets_config.exists(), f"{datasets_config} does not exist."

    config.update(**load_yaml(datasets_config))

    global_config = {
        "max_length": config["max_length"],
        "melspec_config": config["melspec_config"],
        "hubert_config": config["hubert_config"],
    }
    os.makedirs(config["binary_folder"], exist_ok=True)
    with open(pathlib.Path(config["binary_folder"]) / "global_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(global_config, file)

    ForcedAlignmentBinarizer(config).process()


if __name__ == "__main__":
    binarize()
