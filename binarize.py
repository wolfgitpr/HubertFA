import os
import pathlib

import click
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from tools.encoder import UnitsEncoder
from tools.get_melspec import MelSpecExtractor
from tools.load_wav import load_wav


class ForcedAlignmentBinarizer:
    def __init__(self, binary_config):
        self.vocab = None

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

        self.get_melspec = MelSpecExtractor(**binary_config['melspec_config'], device=self.device)

        self.hop_size = binary_config['melspec_config']["hop_length"]

        self.unitsEncoder = UnitsEncoder(
            binary_config['hubert_config']["encoder"],
            binary_config['hubert_config']["model_path"],
            binary_config['hubert_config']["sample_rate"],
            binary_config['hubert_config']["hop_size"],
            self.device)

        self.hubert_channel = binary_config['hubert_config']["channel"]

    def get_vocab(self):
        print("Generating vocab...")
        phonemes = []

        if self.extra_phonemes:
            for ph in self.extra_phonemes:
                if '/' in ph:
                    lang, name = ph.split('/', maxsplit=1)
                    if lang not in self.dictionaries:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"unrecognized language name '{lang}'."
                        )
                    if name in phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"short name conflicts with existing tag."
                        )
                phonemes.append(ph)

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
                    phonemes.extend([f"{lang}/{ph}" if ph not in self.ignored_phonemes else ph for ph in _phonemes])

        phonemes = set(phonemes)
        for p in self.ignored_phonemes:
            if p in phonemes:
                phonemes.remove(p)
        phonemes = sorted(phonemes)
        phonemes = ["SP", *phonemes]

        self.merged_phoneme_groups.insert(0, ["SP", *self.ignored_phonemes])

        vocab = dict(zip(phonemes, range(len(phonemes))))  # phoneme: phoneme_id

        for i, merged_phoneme_group in enumerate(self.merged_phoneme_groups):
            vocab.update({ph: i for ph in merged_phoneme_group})

        for ph in phonemes:
            if ph not in vocab:
                vocab[ph] = len(vocab)

        vocab_dict = {"vocab": vocab,
                      "vocab_size": len(phonemes),
                      "ignored_phonemes": ["SP", *self.ignored_phonemes],
                      "merged_phoneme_groups": self.merged_phoneme_groups,
                      }

        print(f"vocab_size is {len(phonemes)}")

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
        if self.valid_set_size == 0:
            meta_data_valid = (
                meta_data_df[meta_data_df["label_type"] == "full_label"]
                .sample(frac=1)
                .iloc[:valid_set_size, :]
            )
        else:
            meta_data_valid = meta_data_df[meta_data_df["validation"] == True]

        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)
        meta_data_valid = meta_data_valid.reset_index(drop=True)
        meta_data_evaluate = meta_data_evaluate.reset_index(drop=True)

        # binarize valid set
        self.binarize("evaluate", meta_data_evaluate, self.vocab, self.binary_folder)

        # binarize valid set
        self.binarize("valid", meta_data_valid, self.vocab, self.binary_folder)

        # binarize train set
        self.binarize("train", meta_data_train, self.vocab, self.binary_folder)

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

    def make_input_feature(self, wav_path):
        waveform = load_wav(wav_path, self.device, self.sample_rate)  # (L,)

        wav_length = len(waveform) / self.sample_rate  # seconds
        if wav_length > self.max_length:
            print(
                f"Item {wav_path} has a length of {wav_length}s, which is too long, skip it."
            )
            return None, None, None

        # units encode
        units = self.unitsEncoder.encode(waveform.unsqueeze(0), self.sample_rate, self.hop_size)  # [B, C, T]
        melspec = self.get_melspec(waveform)  # [B, C, T]

        B, C, T = units.shape
        if C != self.hubert_channel:
            raise f"Item {wav_path} has unexpect channel of {C}, which should be {self.hubert_channel}."

        return units, melspec, wav_length

    def binarize(
            self,
            prefix: str,
            meta_data: pd.DataFrame,
            vocab: dict,
            binary_data_folder: str | pathlib.Path,
    ):
        print(f"Binarizing {prefix} set...")

        h5py_file_path = pathlib.Path(binary_data_folder) / (prefix + ".h5py")
        h5py_file = h5py.File(h5py_file_path, "w")
        h5py_meta_data = h5py_file.create_group("meta_data")
        items_meta_data = {"label_types": [], "wav_lengths": []}
        h5py_items = h5py_file.create_group("items")

        label_type_to_id = {"no_label": 0, "weak_label": 1, "full_label": 2, "evaluate": 3}

        idx = 0
        total_time = 0.0
        for _, item in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):
            try:
                # input_feature: [1, C, T]
                if not os.path.exists(item["wav_path"]):
                    continue

                units, melspec, wav_length = self.make_input_feature(item.wav_path)

                if units is None:
                    print(f"{wav_length} extract units failed, skip it.")
                    continue

                # label_type: []
                label_type_id = label_type_to_id[item.label_type]
                if label_type_id >= 2:
                    if len(item.ph_dur) != len(item.ph_id_seq):
                        label_type_id = 1
                    if len(item.ph_id_seq) == 0:
                        label_type_id = 0

                ph_id_seq, ph_edge, ph_frame, ph_mask, ph_time = self.make_ph_data(vocab, units.shape[-1],
                                                                                   label_type_id,
                                                                                   item.ph_id_seq,
                                                                                   item.ph_dur)

                if ph_id_seq is None:
                    continue

                ph_seq_raw = [ph for ph in item.ph_seq]
                ph_seq = [ph for ph in ph_seq_raw if vocab["vocab"][ph] != 0]
                assert len(ph_seq) == len(ph_id_seq), "len(ph_seq) != len(ph_id_seq)"

                h5py_item_data = h5py_items.create_group(str(idx))
                idx += 1
                total_time += wav_length
                items_meta_data["wav_lengths"].append(wav_length)
                items_meta_data["label_types"].append(label_type_id)
                h5py_item_data.create_dataset('name', data=str(item["name"]), dtype=h5py.string_dtype(encoding="utf-8"))
                h5py_item_data["input_feature"] = units.cpu().numpy().astype("float32")
                h5py_item_data["melspec"] = melspec.cpu().numpy().astype("float32")
                h5py_item_data["label_type"] = label_type_id
                h5py_item_data.create_dataset('ph_seq_raw', data=ph_seq_raw, dtype=h5py.string_dtype(encoding="utf-8"))
                h5py_item_data.create_dataset('ph_seq', data=ph_seq, dtype=h5py.string_dtype(encoding="utf-8"))
                h5py_item_data["ph_id_seq"] = ph_id_seq.astype("int32")
                h5py_item_data["ph_edge"] = ph_edge.astype("float32")
                h5py_item_data["ph_frame"] = ph_frame.astype("int32")
                h5py_item_data["ph_mask"] = ph_mask.astype("int32")
                h5py_item_data["ph_time"] = ph_time.astype("float32")
            except Exception as e:
                print(f"Failed to binarize: {item}: {e}")

        for k, v in items_meta_data.items():
            h5py_meta_data[k] = np.array(v)
        h5py_file.close()

        len_types = 1 if len(items_meta_data["label_types"]) == 0 else len(items_meta_data["label_types"])

        full_label_ratio = (items_meta_data["label_types"].count(2) + items_meta_data["label_types"].count(
            3)) / len_types
        weak_label_ratio = items_meta_data["label_types"].count(1) / len_types
        no_label_ratio = items_meta_data["label_types"].count(0) / len_types

        print(
            "Data compression ratio: \n"
            f"    full label data: {100 * full_label_ratio:.2f} %,\n"
            f"    weak label data: {100 * weak_label_ratio:.2f} %,\n"
            f"    no label data: {100 * no_label_ratio:.2f} %."
        )
        print(
            f"Successfully binarized {prefix} set, "
            f"total time {total_time:.2f}s ({(total_time / 3600):.2f}h), saved to {h5py_file_path}"
        )

    def get_meta_data(self):
        print("Loading metadata...")
        meta_data_df = pd.DataFrame()
        for dataset in self.datasets:
            language = dataset["language"]
            label_type = dataset["label_type"]
            raw_data_dir = dataset["raw_data_dir"]
            test_prefixes = dataset["test_prefixes"]

            assert label_type in ["full", "weak", "evaluate"], f"{label_type} not in ['full', 'weak', 'evaluate']."
            tuple_prefixes = tuple([x for x in test_prefixes if x])

            csv_path = pathlib.Path(raw_data_dir) / "transcriptions.csv"
            wav_folder = pathlib.Path(raw_data_dir) / "wavs"
            if not os.path.exists(csv_path) or not os.path.exists(wav_folder):
                raise f"{csv_path.absolute()} or {wav_folder.absolute()} does not exist."

            df = pd.read_csv(csv_path, dtype=str)
            if "ph_seq" not in df.columns:
                raise f"{csv_path.absolute()} does not contain 'ph_seq'."
            if label_type == "full" and "ph_dur" not in df.columns:
                raise f"full label csv: {csv_path.absolute()} does not contain 'ph_dur'."

            df["label_type"] = label_type
            df["wav_path"] = df["name"].apply(lambda name: str(wav_folder / (str(name) + ".wav")))
            df["validation"] = df["name"].apply(lambda name: name.startswith(tuple_prefixes))

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
    "--config_path",
    "-c",
    type=str,
    default="configs/binarize_config.yaml",
    show_default=True,
    help="binarize config path",
)
def binarize(config_path: str):
    config = load_yaml(config_path)

    datasets_config = pathlib.Path(config["datasets_config"]).absolute()
    assert os.path.exists(datasets_config), f"{datasets_config} does not exist."

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
