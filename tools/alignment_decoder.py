import numba
import numpy as np
import torch

from tools.plot import plot_for_valid


class AlignmentDecoder:
    def __init__(self, vocab, melspec_config):
        self.vocab = vocab
        self.melspec_config = melspec_config
        self.frame_length = self.melspec_config["hop_length"] / (self.melspec_config["sample_rate"])

        self.ctc_logits = None

        self.ph_seq_id = None
        self.ph_idx_seq = None
        self.ph_frame_pred = None
        self.ph_time_int_pred = None
        self.ph_intervals_pred = None

        self.edge_prob = None
        self.ph_pred_seq = None
        self.frame_confidence = None

    def decode(self,
               ph_frame_logits,
               ph_edge_logits,
               ctc_logits,
               wav_length: float | None,
               ph_seq: list[str],
               word_seq: list[str] = None,
               ph_idx_to_word_idx: list[int] = None,
               ignore_sp: bool = True,
               ):
        ph_seq_id = np.array([self.vocab["vocab"][ph] for ph in ph_seq])
        self.ph_seq_id = ph_seq_id
        ph_mask = np.zeros(self.vocab["vocab_size"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1  # ignored phonemes
        ph_mask = torch.from_numpy(ph_mask)
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        if wav_length is not None:
            num_frames = int(
                (wav_length * self.melspec_config["sample_rate"] + 0.5) / self.melspec_config["hop_length"])
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ph_edge_logits = ph_edge_logits[:, :num_frames]
            ctc_logits = ctc_logits[:, :num_frames, :]

        # [1, 1, vocab_size] unused phonemes inf
        ph_mask = ph_mask.to(ph_frame_logits.device).unsqueeze(0).unsqueeze(0).logical_not() * 1e9

        # [T, vocab_size]
        ph_frame_pred = (
            torch.nn.functional.softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1).squeeze(0)
            .cpu().numpy().astype("float32")
        )

        # [T, vocab_size]
        ph_prob_log = (
            torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1).squeeze(0)
            .cpu().numpy().astype("float32")
        )

        # [T]
        ph_edge_pred = (
            ((torch.nn.functional.sigmoid(ph_edge_logits.float())).clamp(0.0, 1.0)).squeeze(0)
            .cpu().numpy().astype("float32")
        )

        self.ph_frame_pred = ph_frame_pred

        # [1, T, vocab_size]
        self.ctc_logits = (
            ctc_logits.float().squeeze(0).cpu().numpy().astype("float32")
        )  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)  # [T]
        edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)  # [T]

        self.edge_prob = edge_prob

        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,  # [T]
        ) = self._decode(
            ph_seq_id,  # [ph_seq_len]
            ph_prob_log,  # [T, vocab_size]
            edge_prob,  # [T]
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        self.ph_idx_seq = ph_idx_seq
        self.ph_time_int_pred = ph_time_int_pred
        self.frame_confidence = frame_confidence

        # postprocess
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = self.frame_length * (
            np.concatenate(
                [
                    ph_time_int_pred.astype("float32") + ph_time_fractional,
                    [T],
                ]
            )
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        ph_seq_pred = []
        ph_intervals_pred = []
        word_seq_pred = []
        word_intervals_pred = []

        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP" and ignore_sp:
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[i, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[i, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[i, 0], ph_intervals[i, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        self.ph_pred_seq = ph_seq_pred
        self.ph_intervals_pred = ph_intervals_pred

        return ph_seq_pred, ph_intervals_pred, word_seq_pred, word_intervals_pred, total_confidence

    def ctc(self):
        ctc = np.argmax(self.ctc_logits, axis=-1)
        ctc_index = np.concatenate([[0], ctc])
        ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
        ctc = ctc[ctc_index]
        return np.array([ph_id for ph_id in ctc if ph_id != 0])

    def plot(self, melspec, ph_time_gt=None):
        ph_idx_frame = np.zeros(self.ph_frame_pred.shape[0]).astype("int32")
        ph_intervals_pred_int = (
            (self.ph_intervals_pred / self.frame_length).round().astype("int32")
        )
        if ph_time_gt is not None:
            ph_time_gt_int = (
                (ph_time_gt / self.frame_length).round().astype("int32")
            )
        else:
            ph_time_gt_int = None
        last_ph_idx = 0
        for ph_idx, ph_time in zip(self.ph_idx_seq, self.ph_time_int_pred):
            ph_idx_frame[ph_time] += ph_idx - last_ph_idx
            last_ph_idx = ph_idx
        ph_idx_frame = np.cumsum(ph_idx_frame)
        return plot_for_valid(melspec.cpu().numpy(),
                              self.ph_pred_seq,
                              ph_intervals_pred_int,
                              self.frame_confidence,
                              self.ph_frame_pred[:, self.ph_seq_id],
                              ph_idx_frame,
                              self.edge_prob,
                              ph_time_gt_int)

    @staticmethod
    @numba.jit
    def forward_pass(T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s,
                     ph_seq_id,
                     prob3_pad_len):
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + prob_log[t, :] + not_edge_prob_log[t]

            prob2 = np.empty(S, dtype=np.float32)
            prob2[0] = -np.inf
            for i in range(1, S):
                prob2[i] = (
                        dp[t - 1, i - 1]
                        + prob_log[t, i - 1]
                        + edge_prob_log[t]
                        + curr_ph_max_prob_log[i - 1] * (T / S)
                )

            # [t-1,s-2] -> [t,s]
            prob3 = np.empty(S, dtype=np.float32)
            for i in range(prob3_pad_len):
                prob3[i] = -np.inf
            for i in range(prob3_pad_len, S):
                if i - prob3_pad_len + 1 < S - 1 and ph_seq_id[i - prob3_pad_len + 1] != 0:
                    prob3[i] = -np.inf
                else:
                    prob3[i] = (
                            dp[t - 1, i - prob3_pad_len]
                            + prob_log[t, i - prob3_pad_len]
                            + edge_prob_log[t]
                            + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                    )

            stacked_probs = np.empty((3, S), dtype=np.float32)
            for i in range(S):
                stacked_probs[0, i] = prob1[i]
                stacked_probs[1, i] = prob2[i]
                stacked_probs[2, i] = prob3[i]

            for i in range(S):
                max_idx = 0
                max_val = stacked_probs[0, i]
                for j in range(1, 3):
                    if stacked_probs[j, i] > max_val:
                        max_val = stacked_probs[j, i]
                        max_idx = j
                dp[t, i] = max_val
                backtrack_s[t, i] = max_idx

            for i in range(S):
                if backtrack_s[t, i] == 0:
                    curr_ph_max_prob_log[i] = max(curr_ph_max_prob_log[i], prob_log[t, i])
                elif backtrack_s[t, i] > 0:
                    curr_ph_max_prob_log[i] = prob_log[t, i]

            for i in range(S):
                if ph_seq_id[i] == 0:
                    curr_ph_max_prob_log[i] = 0

        return dp, backtrack_s, curr_ph_max_prob_log

    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        # not_SP_num = (ph_seq_id > 0).sum()
        prob_log = ph_prob_log[:, ph_seq_id]

        edge_prob_log = np.log(edge_prob + 1e-6).astype("float32")
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float32")

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
        backtrack_s = np.full_like(dp, -1, dtype="int32")

        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        dp[0, 0] = prob_log[0, 0]
        curr_ph_max_prob_log[0] = prob_log[0, 0]
        if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
            dp[0, 1] = prob_log[0, 1]
            curr_ph_max_prob_log[1] = prob_log[0, 1]

        # forward
        prob3_pad_len = 1
        dp, backtrack_s, curr_ph_max_prob_log = self.forward_pass(
            T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
            prob3_pad_len
        )

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []

        # 如果mode==forced，只能从最后一个音素或者SP结束
        if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
            s = S - 2
        else:
            s = S - 1

        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(
            np.diff(
                np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )
