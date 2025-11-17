import numpy as np

from tools.align_word import WordList, Word
from collections import defaultdict

MIN_SP_LENGTH = 0.1


def add_SP(words_list, wav_length, add_phone="SP"):
    words_res = WordList()
    if words_list[0].start > 0:
        words_res.append(Word(0, words_list[0].start, add_phone, init_phoneme=True))

    words_res.append(words_list[0])
    for i in range(1, len(words_list)):
        word = words_list[i]
        if word.start > words_res[-1].end:
            words_res.append(Word(words_res[-1].end, word.start, add_phone, init_phoneme=True))
        words_res.append(word)

    if words_list[-1].end < wav_length:
        words_res.append(Word(words_list[-1].end, wav_length, add_phone, init_phoneme=True))
    return words_res


def fill_small_gaps(words_list: WordList, wav_length: int):
    if words_list[0].start < 0:
        words_list[0].start = 0

    if words_list[0].start > 0:
        if abs(words_list[0].start) < MIN_SP_LENGTH < words_list[0].dur:
            words_list[0].move_start(0)

    if words_list[-1].end >= wav_length - MIN_SP_LENGTH:
        words_list[-1].move_end(wav_length)

    for i in range(1, len(words_list)):
        if 0 < words_list[i].start - words_list[i - 1].end <= MIN_SP_LENGTH:
            words_list[i].move_start(words_list[i - 1].end)


def post_processing(predictions, add_phone="SP"):
    print("Post-processing...")
    res = []
    error_log = []
    for wav_path, wav_length, words in predictions:
        try:
            fill_small_gaps(words, wav_length)
            words_sp = add_SP(words, wav_length, add_phone)
            res.append([wav_path, wav_length, words_sp])
        except Exception as e:
            error_log.append(f"{wav_path}: {e}")
    return res, error_log


def find_all_duplicate_phonemes(ph_list):
    if len(ph_list) == 1:
        return [0]
    index_dict = defaultdict(list)
    for idx, sublist in enumerate(ph_list):
        key = tuple(sublist)
        index_dict[key].append(idx)

    duplicate_phonemes = {key: indices for key, indices in index_dict.items() if len(indices) > 1}

    if not duplicate_phonemes:
        raise Exception("No duplicate groups")

    sorted_groups = sorted(duplicate_phonemes.items(), key=lambda x: (-len(x[1]), -len(x[0])))
    best_key, best_indices = sorted_groups[0]
    return best_indices


def median_abs_deviation(x, axis=0, center=np.median, scale=1.0):
    """
    Compute the median absolute deviation of the data along the given axis.

    This is a NumPy implementation of scipy.stats.median_abs_deviation.
    """
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            scale = 0.6744897501960817
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = np.asarray(x)

    if x.size == 0:
        if axis is None:
            return np.nan
        if axis is not None:
            nan_shape = list(x.shape)
            del nan_shape[axis]
            nan_shape = tuple(nan_shape)
            if nan_shape == ():
                return np.nan
            return np.full(nan_shape, np.nan)
        return np.nan

    contains_nan = np.isnan(x).any()

    if contains_nan:
        if axis is None:
            return np.nan
        else:
            result_shape = list(x.shape)
            if axis is not None:
                del result_shape[axis]
            result = np.full(tuple(result_shape), np.nan)
            return result / scale
    else:
        if axis is None:
            med = center(x)
            mad = np.median(np.abs(x - med))
        else:
            med = center(x, axis=axis)
            med_expanded = np.expand_dims(med, axis=axis)
            mad = np.median(np.abs(x - med_expanded), axis=axis)

    return mad / scale


def remove_outliers_per_position(data_series, threshold=1.5):
    """
    使用中位数绝对偏差(MAD)方法去除离群值
    参数:
        data_series -- 每个位置的时间戳列表
        threshold -- MAD阈值
    返回: 处理后的平均值列表
    """
    processed_values = []
    for position_values in data_series:
        if not position_values:
            processed_values.append(0.0)
            continue
        med = np.median(position_values)
        mad_val = median_abs_deviation(position_values)

        # 处理MAD=0的情况（所有值相同）
        if mad_val == 0:
            processed_values.append(med)
            continue

        # 计算修正Z-score
        z_scores = np.abs((np.array(position_values) - med) / (mad_val * 1.4826))

        # 分离有效值和离群值
        retained_values = []
        filtered_out = []
        for x, z in zip(position_values, z_scores):
            if z <= threshold:
                retained_values.append(x)
            else:
                filtered_out.append(x)

        if len(retained_values) > 0:
            final_value = np.mean(retained_values)
        else:
            final_value = med

        processed_values.append(final_value)
    return processed_values
