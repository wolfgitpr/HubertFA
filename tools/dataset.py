import pathlib

import h5py
import numpy as np
import torch


class IndexedDatasetBuilder:
    def __init__(self, path, prefix):
        self.path = pathlib.Path(path) / f'{prefix}.data'
        self.prefix = prefix
        self.dset = h5py.File(self.path, 'w')
        self.h5_meta = self.dset.create_group("meta_data")
        self.items = self.dset.create_group("items")

        self.counter = 0
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
        self.wav_lengths.append(item["wav_length"])

    def finalize(self):
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
        self.wav_lengths = None
        self.augmentation_indexes = None

        self.binary_data_folder = binary_data_folder
        self.prefix = prefix

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
        non_speech_target = np.array(item["non_speech_target"])
        non_speech_intervals = np.array(item["non_speech_intervals"])
        return input_feature, ph_seq, ph_id_seq, ph_edge, ph_frame, ph_mask, label_type, melspec, ph_time, name, ph_seq_raw, ph_time_raw, non_speech_target, non_speech_intervals


class BinningAudioBatchSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            wav_lengths: list[int],
            max_length: int = 100,
            binning_length: int = 1000,
            drop_last: bool = False
    ):
        super().__init__()
        assert len(wav_lengths) > 0
        assert max_length > 0
        assert binning_length > 0

        self.max_length = max_length
        self.drop_last = drop_last

        sorted_indices = np.argsort(wav_lengths)[::-1]
        sorted_lengths = np.array(wav_lengths)[sorted_indices]

        self.bins = []
        curr_bin_start_index = 0
        curr_bin_max_length = sorted_lengths[0]

        for i in range(1, len(sorted_lengths)):
            bin_size = i - curr_bin_start_index
            bin_capacity = curr_bin_max_length * bin_size

            if bin_capacity > binning_length:
                batch_size = min(
                    max(1, int(self.max_length // curr_bin_max_length)),
                    bin_size
                )

                bin_indices = sorted_indices[curr_bin_start_index:i]

                if self.drop_last:
                    num_batches = bin_size // batch_size
                else:
                    num_batches = (bin_size + batch_size - 1) // batch_size

                if num_batches > 0:
                    self.bins.append({
                        "indices": bin_indices,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    })

                curr_bin_start_index = i
                curr_bin_max_length = sorted_lengths[i]

        if curr_bin_start_index < len(sorted_lengths):
            bin_size = len(sorted_lengths) - curr_bin_start_index
            batch_size = min(
                max(1, int(self.max_length // curr_bin_max_length)),
                bin_size
            )

            bin_indices = sorted_indices[curr_bin_start_index:]

            if self.drop_last:
                num_batches = bin_size // batch_size
            else:
                num_batches = (bin_size + batch_size - 1) // batch_size

            if num_batches > 0:
                self.bins.append({
                    "indices": bin_indices,
                    "batch_size": batch_size,
                    "num_batches": num_batches
                })

        self.total_batches = sum(bin_info["num_batches"] for bin_info in self.bins)

    def __iter__(self):
        all_batches = []
        for bin_info in self.bins:
            indices = bin_info["indices"]
            batch_size = bin_info["batch_size"]
            num_batches = bin_info["num_batches"]

            np.random.shuffle(indices)

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                all_batches.append(indices[start_idx:end_idx].copy())

        np.random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self.total_batches


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
        non_speech_target = torch.nn.functional.pad(
            torch.as_tensor(item[12]),
            (0, max_len - item[12].shape[-1]),
            mode='constant',
            value=0
        )

        ph_seq = item[1]
        ph_mask = torch.as_tensor(item[5])
        label_type = item[6]
        name = item[9]
        ph_seq_raw = item[10]
        ph_time_raw = item[11]
        non_speech_interval = item[13]

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
            non_speech_target,
            non_speech_interval,
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
    non_speech_target = torch.cat([x[12] for x in padded_batch], dim=0)  # (B, N, T)
    non_speech_intervals = [x[13] for x in padded_batch]  # (B, N, T)

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
        non_speech_target,
        non_speech_intervals
    )
