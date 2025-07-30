import warnings
from dataclasses import dataclass, field


@dataclass
class Phoneme:
    start: float
    end: float
    text: str


@dataclass
class Word:
    start: float
    end: float
    text: str
    phonemes: list[Phoneme] = field(default_factory=list)

    def __init__(self, start: float, end: float, text: str, init_phoneme: bool = False):
        self.start = start
        self.end = end
        self.text = text
        self.phonemes: list[Phoneme] = []
        if init_phoneme:
            self.phonemes.append(Phoneme(start, end, text))

    @property
    def dur(self) -> float:
        return self.end - self.start

    def add_phoneme(self, phoneme):
        if phoneme.start >= self.start and phoneme.end <= self.end:
            self.phonemes.append(phoneme)
        else:
            warnings.warn("phoneme边界超出word，添加失败")

    def append_phoneme(self, phoneme):
        if len(self.phonemes) == 0:
            if phoneme.start == self.start:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warnings.warn("phoneme左边界超出word，添加失败")
        else:
            if phoneme.start == self.phonemes[-1].end:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warnings.warn("phoneme添加失败")

    def move_start(self, new_start):
        if 0 <= new_start < self.phonemes[0].end:
            self.start = new_start
            self.phonemes[0].start = new_start
        else:
            warnings.warn("start >= first_phone_end，无法调整word边界")

    def move_end(self, new_end):
        if new_end > self.phonemes[-1].start >= 0:
            self.end = new_end
            self.phonemes[-1].end = new_end
        else:
            warnings.warn("new_end <= first_phone_start，无法调整word边界")


class WordList(list):
    def overlapping_words(self, new_word: Word):
        overlapping_words = []
        for word in self:
            if not isinstance(word, Word):
                continue
            if word.start < new_word.start < word.end or word.start < new_word.end < word.end:
                overlapping_words.append(word)
        return overlapping_words

    def append(self, word: Word):
        if not isinstance(word, Word):
            raise TypeError("只能添加Word对象")

        if len(word.phonemes) == 0:
            warnings.warn("phones为空，非法word")
            return

        if len(self) == 0:
            super().append(word)
            return

        if not self.overlapping_words(word):
            super().append(word)
        else:
            warnings.warn("区间重叠，无法添加word")

    def add_AP(self, ap: Word, min_dur=0.1):
        if not isinstance(ap, Word):
            raise TypeError("只能添加Word对象")

        if len(ap.phonemes) == 0:
            warnings.warn("phones为空，非法word")
            return

        if len(self) == 0:
            self.append(ap)
            return

        if not self.overlapping_words(ap):
            self.append(ap)
            self.sort(key=lambda w: w.start)
        else:
            for word in self:
                if ap.start <= word.start < word.end <= ap.end:
                    warnings.warn("AP包括整个word，无法添加")
            for word in self:
                if word.start <= ap.start < word.end:
                    ap.move_start(word.end)
                if word.start < ap.end <= word.end:
                    ap.move_end(word.start)
            if ap.start < ap.end and ap.dur > min_dur and not self.overlapping_words(ap):
                self.append(ap)
                self.sort(key=lambda w: w.start)

    @property
    def phonemes(self):
        phonemes = []
        for word in self:
            phonemes.extend([ph.text for ph in word.phonemes])
        return phonemes

    @property
    def intervals(self):
        return [[word.start, word.end] for word in self]

    def clear_language_prefix(self):
        for word in self:
            for phoneme in word.phonemes:
                phoneme.text = phoneme.text.split("/")[-1]
