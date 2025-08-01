import numpy as np

from tools.align_word import Phoneme, Word, WordList
from tools.plot import plot_prob_to_image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return x - x_max - log_sum_exp


class AlignmentDecoder:
    def __init__(self, vocab, class_names, melspec_config):
        self.vocab = vocab
        self.non_speech_phs: list[str] = class_names
        self.melspec_config = melspec_config
        self.frame_length = self.melspec_config["hop_length"] / (self.melspec_config["sample_rate"])

        self.ph_seq_id = None
        self.ph_idx_seq = None
        self.ph_frame_pred = None
        self.ph_time_int_pred = None

        self.ph_seq_pred = None
        self.ph_intervals_pred = None

        self.cvnt_probs = None
        self.edge_prob = None
        self.pred_words = None
        self.frame_confidence = None

    def decode(self,
               ph_frame_logits,
               ph_edge_logits,
               cvnt_logits,
               wav_length: float | None,
               ph_seq: list[str],
               word_seq: list[str] = None,
               ph_idx_to_word_idx: list[int] = None,
               ignore_sp: bool = True,
               non_speech_phonemes: list[str] = None,
               ):
        non_speech_phonemes = non_speech_phonemes or []
        ph_seq_id = np.array([self.vocab["vocab"][ph] for ph in ph_seq])
        self.ph_seq_id = ph_seq_id

        ph_mask = np.full(self.vocab["vocab_size"], 1e9)
        ph_mask[ph_seq_id] = 0
        ph_mask[0] = 0

        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        if wav_length is not None:
            num_frames = int(
                (wav_length * self.melspec_config["sample_rate"] + 0.5) / self.melspec_config["hop_length"])
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ph_edge_logits = ph_edge_logits[:, :num_frames]
            cvnt_logits = cvnt_logits[:, :, :num_frames]

        ph_frame_logits_adjusted = ph_frame_logits - ph_mask[np.newaxis, np.newaxis, :]  # [1, 1, vocab_size]
        ph_frame_pred = softmax(ph_frame_logits_adjusted, axis=-1)[0].astype("float32")  # [T, vocab_size]
        ph_prob_log = log_softmax(ph_frame_logits_adjusted, axis=-1)[0].astype("float32")  # [T, vocab_size]
        ph_edge_pred = np.clip(sigmoid(ph_edge_logits), 0.0, 1.0)[0].astype("float32")  # [T]
        self.ph_frame_pred = ph_frame_pred

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
        self.ph_time_int_pred = np.array(ph_time_int_pred, dtype="int32")
        self.frame_confidence = frame_confidence

        # postprocess
        ph_time_fractional = (edge_diff[self.ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = self.frame_length * (
            np.concatenate([self.ph_time_int_pred.astype("float32") + ph_time_fractional, [T]])
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        word = None
        words: WordList = WordList()

        ph_seq_decoded = []
        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            ph_seq_decoded.append(ph_seq[ph_idx])
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP" and ignore_sp:
                continue
            phoneme = Phoneme(ph_intervals[i, 0], ph_intervals[i, 1], ph_seq[ph_idx])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word.append_phoneme(phoneme)
            else:
                word = Word(ph_intervals[i, 0], ph_intervals[i, 1], word_seq[word_idx])
                word.add_phoneme(phoneme)
                words.append(word)
                word_idx_last = word_idx

        ph_time_pred_int = np.concatenate([self.ph_time_int_pred, [T]])
        ph_intervals_int = np.column_stack([ph_time_pred_int[:-1], ph_time_pred_int[1:]])
        self.ph_seq_pred = words.phonemes
        self.ph_intervals_pred = words.intervals

        speech_phonemes_mask = np.zeros(T, dtype=bool)
        for i, (start, end) in enumerate(ph_intervals_int):
            if ph_seq_decoded[i] != "SP":
                speech_phonemes_mask[start:end] = True

        cvnt_logits[:, 1:, speech_phonemes_mask] = 0
        self.cvnt_probs = softmax(cvnt_logits, axis=1)[0]
        for ph in non_speech_phonemes:
            i = self.non_speech_phs.index(ph)
            tag_words: list[Word] = self.non_speech_words(self.cvnt_probs[i], tag=ph)
            for tag_word in tag_words:
                words.add_AP(tag_word)
        self.pred_words = words
        return words, total_confidence

    def plot(self, melspec, ph_time_gt=None):
        ph_idx_frame = np.zeros(self.ph_frame_pred.shape[0]).astype("int32")
        ph_intervals_pred_int = (
            (np.array(self.ph_intervals_pred) / self.frame_length).round().astype("int32")
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
        return plot_prob_to_image(melspec=melspec,
                                  ph_seq=self.pred_words.phonemes,
                                  ph_intervals=ph_intervals_pred_int,
                                  frame_confidence=self.frame_confidence,
                                  cvnt_prob=self.cvnt_probs,
                                  ph_time_gt=ph_time_gt_int,
                                  label=self.non_speech_phs,
                                  frame_duration=self.frame_length,
                                  )

    @staticmethod
    def forward_pass(T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id, prob3_pad_len=2):
        backtrack_s = np.full_like(dp, -1, dtype=np.int32)
        edge_prob_log = np.log(edge_prob + 1e-6)
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6)
        mask_reset = (ph_seq_id == 0)

        # 预分配数组
        prob1 = np.empty(S, dtype=np.float32)
        prob2 = np.full(S, -np.inf, dtype=np.float32)
        prob3 = np.full(S, -np.inf, dtype=np.float32)

        # 预计算prob3索引和掩码
        i_vals_prob3 = np.arange(prob3_pad_len, S)
        idx_arr = np.clip(i_vals_prob3 - prob3_pad_len + 1, 0, S - 1)
        mask_cond_prob3 = (idx_arr >= S - 1) | (ph_seq_id[idx_arr] == 0)

        for t in range(1, T):
            prob_log_t = prob_log[t]
            edge_log_t = edge_prob_log[t]
            not_edge_log_t = not_edge_prob_log[t]
            dp_prev = dp[t - 1]

            # 类型1转移: 停留在当前音素
            prob1[:] = dp_prev + prob_log_t + not_edge_log_t

            # 类型2转移: 移动到下一个音素
            prob2[1:] = (
                    dp_prev[:S - 1] +
                    prob_log_t[:S - 1] +
                    edge_log_t +
                    curr_ph_max_prob_log[:S - 1] * (T / S)
            )

            # 类型3转移: 跳转到后续音素
            candidate_vals = (
                    dp_prev[:S - prob3_pad_len] +
                    prob_log_t[:S - prob3_pad_len] +
                    edge_log_t +
                    curr_ph_max_prob_log[:S - prob3_pad_len] * (T / S)
            )
            prob3[i_vals_prob3] = np.where(
                mask_cond_prob3, candidate_vals, -np.inf
            )

            # 组合概率并找出最佳转移
            stacked_probs = np.vstack((prob1, prob2, prob3))
            max_indices = np.argmax(stacked_probs, axis=0)
            dp[t] = stacked_probs[max_indices, np.arange(S)]
            backtrack_s[t] = max_indices

            # 更新当前音素最大概率
            mask_type0 = (max_indices == 0)
            mask_type12 = ~mask_type0
            np.maximum(
                curr_ph_max_prob_log,
                prob_log_t,
                out=curr_ph_max_prob_log,
                where=mask_type0
            )
            np.copyto(
                curr_ph_max_prob_log,
                prob_log_t,
                where=mask_type12
            )
            curr_ph_max_prob_log[mask_reset] = 0.0

            # 重置临时数组
            prob2[1:] = -np.inf
            prob3[i_vals_prob3] = -np.inf

        return dp, backtrack_s, curr_ph_max_prob_log

    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        prob_log = ph_prob_log[:, ph_seq_id]

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)

        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        dp[0, 0] = prob_log[0, 0]
        curr_ph_max_prob_log[0] = prob_log[0, 0]
        if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
            dp[0, 1] = prob_log[0, 1]
            curr_ph_max_prob_log[1] = prob_log[0, 1]

        # forward
        dp, backtrack_s, curr_ph_max_prob_log = self.forward_pass(
            T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id,
            prob3_pad_len=2 if S >= 2 else 1
        )

        # backward
        ph_idx_seq, ph_time_int, frame_confidence = [], [], []

        # 如果mode==forced，只能从最后一个音素或者SP结束
        s = S - 2 if dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0 else S - 1

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

        return ph_idx_seq, ph_time_int, frame_confidence

    def non_speech_words(self, prob, threshold=0.5, max_gap=5, ap_threshold=10, tag=""):
        """
        Find segments in the array where values are mostly above a given threshold.

        :param tag: phoneme text
        :param ap_threshold: minimum breathing time, unit: number of samples
        :param prob: numpy array of probabilities
        :param threshold: threshold value to consider a segment
        :param max_gap: maximum allowed gap of values below the threshold within a segment
        :return: list of tuples (start_index, end_index) of segments
        """
        words = []
        start = None
        gap_count = 0

        for i in range(len(prob)):
            if prob[i] >= threshold:
                if start is None:
                    start = i
                gap_count = 0
            else:
                if start is not None:
                    if gap_count < max_gap:
                        gap_count += 1
                    else:
                        end = i - gap_count - 1
                        if end > start and (end - start) >= ap_threshold:
                            word = Word(start * self.frame_length, end * self.frame_length, tag)
                            word.add_phoneme(Phoneme(start * self.frame_length, end * self.frame_length, tag))
                            words.append(word)
                        start = None
                        gap_count = 0

        # Handle the case where the array ends with a segment
        if start is not None and (len(prob) - start) >= ap_threshold:
            word = Word(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag)
            word.add_phoneme(Phoneme(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag))
            words.append(word)
        return words
