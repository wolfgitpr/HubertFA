import pathlib

import h5py
import numpy as np
import pandas as pd
import torch


class IndexedDatasetBuilder:
    def __init__(self, path, prefix):
        self.path = pathlib.Path(path) / f'{prefix}.data'
        self.prefix = prefix
        self.dset = h5py.File(self.path, 'w')
        self.h5_meta = self.dset.create_group("meta_data")
        self.items = self.dset.create_group("items")

        self.counter = 0

        self.label_types = []
        self.wav_lengths = []

    def add_item(self, item):
        if item is None:
            return
        group = self.items.create_group(str(self.counter))
        self.counter += 1
        for key, value in item.items():
            if key in ['name', 'ph_seq', 'ph_seq_raw']:
                group.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
            else:
                group[key] = value

        self.label_types.append(item["label_type"])
        self.wav_lengths.append(item["wav_length"])

    def finalize(self):
        self.h5_meta.create_dataset("label_types", data=np.array(self.label_types, dtype=np.int32))
        self.h5_meta.create_dataset("wav_lengths", data=np.array(self.wav_lengths, dtype=np.float32))
        self.dset.close()


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            binary_data_folder="data/binary",
            prefix="train",
    ):
        # do not open hdf5 here
        self.h5py_file = None
        self.label_types = None
        self.wav_lengths = None
        self.augmentation_indexes = None

        self.binary_data_folder = binary_data_folder
        self.prefix = prefix

    def get_label_types(self):
        uninitialized = self.label_types is None
        if uninitialized:
            self._open_h5py_file()
        ret = self.label_types
        if uninitialized:
            self._close_h5py_file()
        return ret

    def get_wav_lengths(self):
        uninitialized = self.wav_lengths is None
        if uninitialized:
            self._open_h5py_file()
        ret = self.wav_lengths
        if uninitialized:
            self._close_h5py_file()
        return ret

    def _open_h5py_file(self):
        self.h5py_file = h5py.File(
            str(pathlib.Path(self.binary_data_folder) / (self.prefix + ".data")), "r"
        )
        self.label_types = np.array(self.h5py_file["meta_data"]["label_types"])
        self.wav_lengths = np.array(self.h5py_file["meta_data"]["wav_lengths"])

    def _close_h5py_file(self):
        self.h5py_file.close()
        self.h5py_file = None

    def __len__(self):
        uninitialized = self.h5py_file is None
        if uninitialized:
            self._open_h5py_file()
        ret = len(self.h5py_file["items"])
        if uninitialized:
            self._close_h5py_file()
        return ret

    def __getitem__(self, index):
        if self.h5py_file is None:
            self._open_h5py_file()

        item = self.h5py_file["items"][str(index)]
        name = item["name"][()].decode('utf-8')
        input_feature = np.array(item["input_feature"])  # [1,256,T]
        label_type = np.array(item["label_type"])
        ph_seq_raw = [ph.decode('utf-8') for ph in item["ph_seq_raw"]]
        ph_seq = [ph.decode('utf-8') for ph in item["ph_seq"]]
        ph_id_seq = np.array(item["ph_id_seq"])
        ph_edge = np.array(item["ph_edge"])
        ph_frame = np.array(item["ph_frame"])
        ph_mask = np.array(item["ph_mask"])
        melspec = np.array(item["melspec"])
        ph_time = np.array(item["ph_time"])
        ph_time_raw = np.array(item["ph_time_raw"])

        return input_feature, ph_seq, ph_id_seq, ph_edge, ph_frame, ph_mask, label_type, melspec, ph_time, name, ph_seq_raw, ph_time_raw


class WeightedBinningAudioBatchSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            type_ids,
            wav_lengths,
            oversampling_weights=None,
            max_length=100,
            binning_length=1000,
            drop_last=False,
    ):
        if oversampling_weights is None:
            oversampling_weights = [1] * (max(type_ids) + 1)
        oversampling_weights = np.array(oversampling_weights).astype(np.float32)

        assert min(oversampling_weights) > 0
        assert len(oversampling_weights) >= max(type_ids) + 1
        assert min(type_ids) >= 0
        assert len(type_ids) == len(wav_lengths)
        assert max_length > 0
        assert binning_length > 0

        count = np.bincount(type_ids)
        count = np.pad(count, (0, len(oversampling_weights) - len(count)))
        self.oversampling_weights = oversampling_weights / min(
            oversampling_weights[count > 0]
        )
        self.max_length = max_length
        self.drop_last = drop_last

        # sort by wav_lengths
        meta_data = (
            pd.DataFrame(
                {
                    "dataset_index": range(len(type_ids)),
                    "type_id": type_ids,
                    "wav_length": wav_lengths,
                }
            )
            .sort_values(by=["wav_length"], ascending=False)
            .reset_index(drop=True)
        )

        # binning and compute oversampling num
        self.bins = []

        curr_bin_start_index = 0
        curr_bin_max_item_length = meta_data.loc[0, "wav_length"]
        for i in range(len(meta_data)):
            if curr_bin_max_item_length * (i - curr_bin_start_index) > binning_length:
                bin_data = {
                    "batch_size": self.max_length // curr_bin_max_item_length,
                    "num_batches": 0,
                    "type": [],
                }

                item_num = 0
                for type_id, weight in enumerate(self.oversampling_weights):
                    idx_list = (
                        meta_data.loc[curr_bin_start_index: i - 1]
                        .loc[meta_data["type_id"] == type_id]
                        .to_dict(orient="list")["dataset_index"]
                    )

                    oversample_num = np.round(len(idx_list) * (weight - 1))
                    bin_data["type"].append(
                        {
                            "idx_list": idx_list,
                            "oversample_num": oversample_num,
                        }
                    )
                    item_num += len(idx_list) + oversample_num

                if bin_data["batch_size"] <= 0:
                    raise ValueError(
                        "batch_size <= 0, maybe batch_max_length in training config is too small "
                        "or max_length in binarizing config is too long."
                    )
                num_batches = item_num / bin_data["batch_size"]
                if self.drop_last:
                    bin_data["num_batches"] = int(num_batches)
                else:
                    bin_data["num_batches"] = int(np.ceil(num_batches))
                self.bins.append(bin_data)

                curr_bin_start_index = i
                curr_bin_max_item_length = meta_data.loc[i, "wav_length"]

        self.len = None

    def __len__(self):
        if self.len is None:
            self.len = 0
            for bin_data in self.bins:
                self.len += bin_data["num_batches"]
        return self.len

    def __iter__(self):
        np.random.shuffle(self.bins)

        for bin_data in self.bins:
            batch_size = bin_data["batch_size"]
            num_batches = bin_data["num_batches"]

            idx_list = []
            for type_id, weight in enumerate(self.oversampling_weights):
                idx_list_of_type = bin_data["type"][type_id]["idx_list"]
                oversample_num = bin_data["type"][type_id]["oversample_num"]

                if len(idx_list_of_type) > 0:
                    idx_list.extend(idx_list_of_type)
                    oversample_idx_list = np.random.choice(
                        idx_list_of_type, int(oversample_num)
                    )
                    idx_list.extend(oversample_idx_list)

            idx_list = np.random.permutation(idx_list)

            if self.drop_last:
                num_batches = int(num_batches)
                idx_list = idx_list[: num_batches * batch_size]
            else:
                num_batches = int(np.ceil(num_batches))
                random_idx = np.random.choice(
                    idx_list, int(num_batches * batch_size - len(idx_list))
                )
                idx_list = np.concatenate([idx_list, random_idx])

            np.random.shuffle(idx_list)

            for i in range(num_batches):
                yield idx_list[int(i * batch_size): int((i + 1) * batch_size)]


def collate_fn(batch):
    """Collate function for processing a batch of data samples.

    Args:
        batch (list of tuples): Each tuple contains elements from MixedDataset:
            input_feature, ph_seq, ph_edge, ph_frame, ph_mask, label_type, melspec.

    Returns:
        input_feature: (B T C)
        input_feature_lengths: (B)
        ph_seq: (B S)
        ph_seq_lengths: (B)
        ph_edge: (B T)
        ph_frame: (B T)
        ph_mask: (B vocab_size)
        label_type: (B)
        melspec: (B T)
    """
    # Calculate maximum lengths for padding
    input_feature_lengths = torch.tensor([item[0].shape[-2] for item in batch])
    max_len = input_feature_lengths.max().item()
    ph_seq_lengths = torch.tensor([len(item[1]) for item in batch])
    max_ph_seq_len = ph_seq_lengths.max().item()

    padded_batch = []
    for item in batch:
        # Pad each element in the sample
        input_feature = torch.nn.functional.pad(
            torch.as_tensor(item[0]),
            (0, 0, 0, max_len - item[0].shape[-2], 0, 0),
            mode='constant',
            value=0
        )
        melspec = torch.nn.functional.pad(
            torch.as_tensor(item[7]),
            (0, max_len - item[7].shape[-1]),
            mode='constant',
            value=0
        )

        ph_id_seq = torch.nn.functional.pad(
            torch.as_tensor(item[2]),
            (0, max_ph_seq_len - len(item[2])),
            mode='constant',
            value=0
        )
        ph_edge = torch.nn.functional.pad(
            torch.as_tensor(item[3]),
            (0, max_len - len(item[3])),
            mode='constant',
            value=0
        )
        ph_frame = torch.nn.functional.pad(
            torch.as_tensor(item[4]),
            (0, max_len - len(item[4])),
            mode='constant',
            value=0
        )
        ph_time = torch.nn.functional.pad(
            torch.as_tensor(item[8]),
            (0, max_ph_seq_len - len(item[8])),
            mode='constant',
            value=0
        )
        ph_seq = item[1]
        ph_mask = torch.as_tensor(item[5])
        label_type = item[6]
        name = item[9]
        ph_seq_raw = item[10]
        ph_time_raw = item[11]

        padded_batch.append((
            input_feature,
            ph_seq,
            ph_id_seq,
            ph_edge,
            ph_frame,
            ph_mask,
            label_type,
            melspec,
            ph_time,
            name,
            ph_seq_raw,
            ph_time_raw,
        ))

    # Concatenate/stack tensors efficiently
    input_features = torch.cat([x[0] for x in padded_batch], dim=0)  # (B, C, T)
    ph_seqs = [x[1] for x in padded_batch]
    ph_id_seqs = torch.stack([x[2] for x in padded_batch])  # (B, S_ph)
    ph_edges = torch.stack([x[3] for x in padded_batch])  # (B, T)
    ph_frames = torch.stack([x[4] for x in padded_batch])  # (B, T)
    ph_masks = torch.stack([x[5] for x in padded_batch])  # (B, ...)
    label_types = torch.tensor(np.array([x[6] for x in padded_batch]))  # (B,)
    melspecs = torch.cat([x[7] for x in padded_batch], dim=0)  # (B, C_mel, T)
    ph_times = torch.stack([x[8] for x in padded_batch])  # (B, S_ph)
    names = [x[9] for x in padded_batch]
    ph_seq_raws = [x[10] for x in padded_batch]
    ph_time_raws = [x[11] for x in padded_batch]

    return (
        input_features,
        input_feature_lengths,
        ph_seqs,
        ph_id_seqs,
        ph_seq_lengths,
        ph_edges,
        ph_frames,
        ph_masks,
        label_types,
        melspecs,
        ph_times,
        names,
        ph_seq_raws,
        ph_time_raws,
    )
