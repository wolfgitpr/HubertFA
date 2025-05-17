from networks.g2p.base_g2p import BaseG2P


class PhonemeG2P(BaseG2P):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _g2p(self, input_text):
        _word_seq = input_text.strip().split(" ")
        _word_seq = [ph for ph in _word_seq if ph != "SP"]
        _ph_seq = ["SP"]
        _ph_idx_to_word_idx = [-1]
        for word_idx, word in enumerate(_word_seq):
            _ph_seq.append(word)
            _ph_idx_to_word_idx.append(word_idx)
            _ph_seq.append("SP")
            _ph_idx_to_word_idx.append(-1)
        return _ph_seq, _word_seq, _ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = PhonemeG2P(**{"language": "zh"})
    text = "wo shi yi ge xue sheng SP SP SP"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
