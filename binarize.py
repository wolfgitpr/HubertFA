import os
import pathlib
import warnings

import click
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from networks.utils.get_melspec import MelSpecExtractor
from networks.utils.load_wav import load_wav
from networks.vocoder.hubert import UnitsEncoder


class ForcedAlignmentBinarizer:
    def __init__(
            self,
            data_folder,
            binary_folder,
            valid_set_size,
            valid_sets,
            valid_set_preferred_folders,
            ignored_phonemes,
            melspec_config,
            max_length,
            dictionary_paths,
            vowel_phonemes,
            hubert_config: dict = None,
    ):
        self.data_folder = pathlib.Path(data_folder)
        self.binary_folder = pathlib.Path(binary_folder)

        self.valid_set_size = valid_set_size
        self.valid_sets = valid_sets
        self.valid_set_preferred_folders = valid_set_preferred_folders

        self.ignored_phonemes = ignored_phonemes
        self.melspec_config = melspec_config

        self.dictionary_paths = dictionary_paths
        self.vowel_phonemes = vowel_phonemes

        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate = self.melspec_config["sample_rate"]
        self.frame_length = self.melspec_config["hop_length"] / self.sample_rate

        self.get_melspec = MelSpecExtractor(**melspec_config, device=self.device)

        self.hop_size = melspec_config["hop_length"]

        self.combine_mel = hubert_config["combine_mel"]
        self.unitsEncoder = UnitsEncoder(
            hubert_config["encoder"],
            hubert_config["model_path"],
            hubert_config["sample_rate"],
            hubert_config["hop_size"],
            self.device)

    @staticmethod
    def get_vocab(data_folder_path, ignored_phonemes):
        print("Generating vocab...")
        phonemes = []
        trans_path_list = data_folder_path.rglob("transcriptions.csv")

        for trans_path in trans_path_list:
            if trans_path.name == "transcriptions.csv":
                df = pd.read_csv(trans_path)
                ph = list(set(" ".join(df["ph_seq"]).split(" ")))
                phonemes.extend(ph)

        phonemes = set(phonemes)
        for p in ignored_phonemes:
            if p in phonemes:
                phonemes.remove(p)
        phonemes = sorted(phonemes)
        phonemes = ["SP", *phonemes]

        vocab = dict(zip(phonemes, range(len(phonemes))))
        vocab.update(dict(zip(range(len(phonemes)), phonemes)))
        vocab.update({i: 0 for i in ignored_phonemes})
        vocab.update({"<vocab_size>": len(phonemes)})

        print(f"vocab_size is {len(phonemes)}")

        return vocab

    @staticmethod
    def get_vowel(dictionary_paths, ignored_phonemes, vowel_phonemes, vocab):
        print("Generating vowel phonemes...")
        vowels = []

        for path in dictionary_paths:
            with open(path, "r", encoding="utf-8") as dict_file:
                for line in dict_file:
                    _, phonemes = line.strip().split('\t')
                    phonemes = phonemes.split(' ')
                    if len(phonemes) == 1:
                        vowels.append(phonemes[0])
                    elif len(phonemes) == 2:
                        vowels.append(phonemes[1])

        for v in vowel_phonemes:
            vowels.append(v)

        vowels = set(vowels)
        for p in ignored_phonemes:
            if p in vowels:
                vowels.remove(p)
        vowels = sorted(vowels)

        vowel_dict = {}
        for v in vowels:
            if v in vocab.keys():
                vowel_dict[v] = vocab[v]

        print(f"vowels_size is {len(vowels)}")

        return vowel_dict

    def process(self):
        vocab = self.get_vocab(self.data_folder, self.ignored_phonemes)
        with open(self.binary_folder / "vocab.yaml", "w") as file:
            yaml.dump(vocab, file)

        vowels = self.get_vowel(self.dictionary_paths, self.ignored_phonemes, self.vowel_phonemes, vocab)
        with open(self.binary_folder / "vowel.yaml", "w") as file:
            yaml.dump(vowels, file)

        # load metadata of each item
        meta_data_df = self.get_meta_data(self.data_folder, vocab)

        # split train and valid set
        valid_set_size = int(self.valid_set_size)
        if len(self.valid_sets) == 0:
            meta_data_valid = (
                meta_data_df[meta_data_df["label_type"] != "no_label"]
                .sample(frac=1)
                .sort_values(by="preferred", ascending=False)
                .iloc[:valid_set_size, :]
            )
        else:
            meta_data_valid = (
                meta_data_df[(meta_data_df["label_type"] != "no_label") & (meta_data_df["name"].isin(self.valid_sets))]
            )
        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(
            drop=True
        )
        meta_data_valid = meta_data_valid.reset_index(drop=True)

        # binarize valid set
        self.binarize(
            "valid",
            meta_data_valid,
            vocab,
            self.binary_folder
        )

        # binarize train set
        self.binarize(
            "train",
            meta_data_train,
            vocab,
            self.binary_folder
        )

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

        label_type_to_id = {"no_label": 0, "weak_label": 1, "full_label": 2}

        idx = 0
        total_time = 0.0
        for _, item in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):
            try:
                # input_feature: [1, input_dim, T]
                if not os.path.exists(item["wav_path"]):
                    continue
                waveform = load_wav(item.wav_path, self.device, self.sample_rate)   # (L,)

                # units encode
                units_t = self.unitsEncoder.encode(waveform.unsqueeze(0), self.sample_rate, self.hop_size)  # [B, T, C]

                input_feature = units_t.transpose(1, 2).squeeze(0)  # [C, T]
                melspec = self.get_melspec(waveform, 0)  # [C, T]

                if self.combine_mel:
                    input_feature = torch.cat([input_feature, melspec], dim=0) # [Units_C + Mel_C, T]

                wav_length = len(waveform) / self.sample_rate   # seconds
                T = input_feature.shape[-1]
                if wav_length > self.max_length:
                    print(
                        f"Item {item.wav_path} has a length of {wav_length}s, which is too long, skip it."
                    )
                    continue
                else:
                    h5py_item_data = h5py_items.create_group(str(idx))
                    items_meta_data["wav_lengths"].append(wav_length)
                    idx += 1
                    total_time += wav_length

                input_feature = input_feature.unsqueeze(0)  # [B, C, T]
                melspec = melspec.unsqueeze(0)  # [B, C, T]

                input_feature = (input_feature - input_feature.mean(dim=[1, 2], keepdim=True)) / input_feature.std(
                    dim=[1, 2], keepdim=True)   # [B, C, T]

                h5py_item_data["input_feature"] = (
                    input_feature.cpu().numpy().astype("float32")
                )

                h5py_item_data["melspec"] = melspec.cpu().numpy().astype("float32")

                # label_type: []
                label_type_id = label_type_to_id[item.label_type]
                if label_type_id == 2:
                    if len(item.ph_dur) != len(item.ph_seq):
                        label_type_id = 1
                    if len(item.ph_seq) == 0:
                        label_type_id = 0
                h5py_item_data["label_type"] = label_type_id
                items_meta_data["label_types"].append(label_type_id)

                if label_type_id == 0:
                    # ph_seq: [S]
                    ph_seq = np.array([]).astype("int32")

                    # ph_edge: [T]
                    ph_edge = np.zeros([T], dtype="float32")

                    # ph_frame: [T]
                    ph_frame = np.zeros(T, dtype="int32")

                    # ph_mask: [vocab_size]
                    ph_mask = np.ones(vocab["<vocab_size>"], dtype="int32")
                elif label_type_id == 1:
                    # ph_seq: [S]
                    ph_seq = np.array(item.ph_seq).astype("int32")
                    ph_seq = ph_seq[ph_seq != 0]

                    # ph_edge: [T]
                    ph_edge = np.zeros([T], dtype="float32")

                    # ph_frame: [T]
                    ph_frame = np.zeros(T, dtype="int32")

                    # ph_mask: [vocab_size]
                    ph_mask = np.zeros(vocab["<vocab_size>"], dtype="int32")
                    ph_mask[ph_seq] = 1
                    ph_mask[0] = 1
                elif label_type_id == 2:
                    # ph_seq: [S]
                    ph_seq = np.array(item.ph_seq).astype("int32")
                    not_sp_idx = ph_seq != 0
                    ph_seq = ph_seq[not_sp_idx]

                    # ph_edge: [T]
                    ph_dur = np.array(item.ph_dur).astype("float32")
                    ph_time = np.array(np.concatenate(([0], ph_dur))).cumsum() / self.frame_length
                    ph_interval = np.stack((ph_time[:-1], ph_time[1:]))

                    ph_interval = ph_interval[:, not_sp_idx]
                    ph_seq = ph_seq
                    ph_time = np.unique(ph_interval.flatten())
                    if ph_time[-1] >= T:
                        ph_time = ph_time[:-1]

                    ph_edge = np.zeros([T], dtype="float32")
                    if len(ph_seq) > 0:
                        if ph_time[-1] + 0.5 > T:
                            ph_time = ph_time[:-1]
                        if ph_time[0] - 0.5 < 0:
                            ph_time = ph_time[1:]
                        ph_time_int = np.round(ph_time).astype("int32")
                        ph_time_fractional = ph_time - ph_time_int

                        ph_edge[ph_time_int] = 0.5 + ph_time_fractional
                        ph_edge[ph_time_int - 1] = 0.5 - ph_time_fractional
                        ph_edge = ph_edge * 0.8 + 0.1

                    # ph_frame: [T]
                    ph_frame = np.zeros(T, dtype="int32")
                    if len(ph_seq) > 0:
                        for ph_id, st, ed in zip(
                                ph_seq, ph_interval[0], ph_interval[1]
                        ):
                            if st < 0:
                                st = 0
                            if ed > T:
                                ed = T
                            ph_frame[int(np.round(st)): int(np.round(ed))] = ph_id

                    # ph_mask: [vocab_size]
                    ph_mask = np.zeros(vocab["<vocab_size>"], dtype="int32")
                    if len(ph_seq) > 0:
                        ph_mask[ph_seq] = 1
                    ph_mask[0] = 1
                else:
                    raise ValueError("Unknown label type.")

                h5py_item_data["ph_seq"] = ph_seq.astype("int32")
                h5py_item_data["ph_edge"] = ph_edge.astype("float32")
                h5py_item_data["ph_frame"] = ph_frame.astype("int32")
                h5py_item_data["ph_mask"] = ph_mask.astype("int32")
            except Exception as e:
                e.args += (item.wav_path,)
                print(e)
                continue
        for k, v in items_meta_data.items():
            h5py_meta_data[k] = np.array(v)
        h5py_file.close()
        full_label_ratio = items_meta_data["label_types"].count(2) / len(
            items_meta_data["label_types"]
        )
        weak_label_ratio = items_meta_data["label_types"].count(1) / len(
            items_meta_data["label_types"]
        )
        no_label_ratio = items_meta_data["label_types"].count(0) / len(
            items_meta_data["label_types"]
        )
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

    def get_meta_data(self, data_folder, vocab):
        full_path = pathlib.Path(os.path.join(data_folder, "full_label"))
        weak_path = pathlib.Path(os.path.join(data_folder, "weak_label"))

        trans_path_list = ([i for i in full_path.rglob("transcriptions.csv") if i.name == "transcriptions.csv"] +
                           [i for i in weak_path.rglob("transcriptions.csv") if i.name == "transcriptions.csv"])
        if len(trans_path_list) <= 0:
            warnings.warn(f"No transcriptions.csv found in {data_folder}.")

        print("Loading metadata...")
        meta_data_df = pd.DataFrame()
        for trans_path in tqdm(trans_path_list):
            df = pd.read_csv(trans_path, dtype=str)
            df["wav_path"] = df["name"].apply(
                lambda name: str(trans_path.parent / "wavs" / (str(name) + ".wav")),
            )
            df["preferred"] = df["wav_path"].apply(
                lambda path_: (
                    True if any([i in pathlib.Path(path_).parts for i in self.valid_set_preferred_folders])
                    else False
                ),
            )
            df["label_type"] = df["wav_path"].apply(
                lambda path_: (
                    "full_label" if "full_label" in path_
                    else "weak_label" if "weak_label" in path_ else "no_label"
                ),
            )
            if len(meta_data_df) >= 1:
                meta_data_df = pd.concat([meta_data_df, df])
            else:
                meta_data_df = df

        no_label_df = pd.DataFrame(
            {"wav_path": [i for i in (data_folder / "no_label").rglob("*.wav")]}
        )
        meta_data_df = pd.concat([meta_data_df, no_label_df])
        meta_data_df["label_type"].fillna("no_label", inplace=True)

        meta_data_df.reset_index(drop=True, inplace=True)

        meta_data_df["ph_seq"] = meta_data_df["ph_seq"].apply(
            lambda x: ([vocab[i] for i in x.split(" ")] if isinstance(x, str) else [])
        )
        if "ph_dur" in meta_data_df.columns:
            meta_data_df["ph_dur"] = meta_data_df["ph_dur"].apply(
                lambda x: (
                    [float(i) for i in x.split(" ")] if isinstance(x, str) else []
                )
            )
        meta_data_df = meta_data_df.sort_values(by="label_type").reset_index(drop=True)

        return meta_data_df


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
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_config = {
        "max_length": config["max_length"],
        "melspec_config": config["melspec_config"]
    }
    os.makedirs(config["binary_folder"], exist_ok=True)
    with open(pathlib.Path(config["binary_folder"]) / "global_config.yaml", "w") as file:
        yaml.dump(global_config, file)

    ForcedAlignmentBinarizer(**config).process()


if __name__ == "__main__":
    binarize()
