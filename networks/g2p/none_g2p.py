import numpy as np

from networks.g2p.base_g2p import BaseG2P


class NoneG2P(BaseG2P):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _g2p(self, input_text):
        input_seq = input_text.strip().split(" ")

        _ph_seq = ["SP"]
        for i, ph in enumerate(input_seq):
            if ph == "SP" and _ph_seq[-1] == "SP":
                continue
            _ph_seq.append(ph)
        if _ph_seq[-1] != "SP":
            _ph_seq.append("SP")

        _word_seq = _ph_seq
        _ph_idx_to_word_idx = np.arange(len(_ph_seq))

        return _ph_seq, _word_seq, _ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = NoneG2P(**{"language": "zh", "non_speech_phonemes": []})
    text = "wo shi SP yi ge xue sheng"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
