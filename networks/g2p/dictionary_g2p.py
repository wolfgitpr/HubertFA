import warnings

from networks.g2p.base_g2p import BaseG2P


class DictionaryG2P(BaseG2P):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dict_path = kwargs["dictionary"]
        with open(dict_path, "r") as f:
            dictionary = f.read().strip().split("\n")
        self.dictionary = {
            item.split("\t")[0].strip(): item.split("\t")[1].strip().split(" ")
            for item in dictionary
        }

    def _g2p(self, input_text):
        word_seq_raw = input_text.strip().split(" ")
        _word_seq = []
        word_seq_idx = 0
        _ph_seq = ["SP"]
        _ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            if word not in self.dictionary:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            _word_seq.append(word)
            phones = self.dictionary[word]
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                _ph_seq.append(ph)
                _ph_idx_to_word_idx.append(word_seq_idx)
            if _ph_seq[-1] != "SP":
                _ph_seq.append("SP")
                _ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return _ph_seq, _word_seq, _ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = DictionaryG2P(
        **{"dictionary": "../../dictionaries/opencpop-extension.txt", "language": "zh", "non_speech_phonemes": []}
    )
    text = "wo SP shi yi ge xue sheng a"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
