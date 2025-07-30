from tools.align_word import WordList, Word

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


def fill_small_gaps(words_list: WordList, wav_length):
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
    for wav_path, wav_length, words, confidence in predictions:
        try:
            fill_small_gaps(words, wav_length)
            words_sp = add_SP(words, wav_length, add_phone)
            res.append([wav_path, wav_length, words_sp, confidence])
        except Exception as e:
            error_log.append(f"{wav_path}: {e}")
    return res, error_log
