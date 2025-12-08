import pathlib
import textgrid


class Exporter:
    def __init__(self, predictions, output_folder=None):
        self.predictions = predictions
        self.output_folder = pathlib.Path(output_folder) if output_folder else None

    def save_textgrids(self):
        print("Saving TextGrids...")
        for wav_path, wav_length, words in self.predictions:
            wav_path = pathlib.Path(wav_path)
            tg = textgrid.TextGrid(minTime=0, maxTime=wav_length)
            word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=wav_length)
            ph_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=wav_length)

            for word in words:
                word_tier.add(minTime=word.start, maxTime=word.end, mark=word.text)
                for phoneme in word.phonemes:
                    ph_tier.add(minTime=max(0, phoneme.start), maxTime=phoneme.end, mark=phoneme.text)

            tg.append(word_tier)
            tg.append(ph_tier)

            if self.output_folder is not None:
                tg_path = self.output_folder / "TextGrid" / wav_path.with_suffix(".TextGrid").name
            else:
                tg_path = wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name

            tg_path.parent.mkdir(parents=True, exist_ok=True)
            tg.write(tg_path)

    def export(self, out_formats):
        if 'textgrid' in out_formats:
            self.save_textgrids()
