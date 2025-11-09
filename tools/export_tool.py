import pathlib

import pandas as pd
import textgrid
# from icecream import ic


class Exporter:
    def __init__(self, predictions, out_path=None):
        self.predictions = predictions
        self.out_path = pathlib.Path(out_path) if out_path else None

    def save_textgrids(self):
        print("Saving TextGrids...")
        for wav_path, wav_length, words, confidence in self.predictions:
            wav_path = pathlib.Path(wav_path)
            # ic(wav_path)

            try:
                tg = textgrid.TextGrid()
                word_tier = textgrid.IntervalTier(name="words", minTime=0.0)
                ph_tier = textgrid.IntervalTier(name="phones", minTime=0.0)

                for word in words:
                    word_tier.add(minTime=word.start, maxTime=word.end, mark=word.text)
                    
                    for phoneme in word.phonemes:
                        startTime = max(0, phoneme.start)
                        endTime = phoneme.end
                        
                        ph_tier.add(minTime=startTime, maxTime=endTime, mark=phoneme.text)

                tg.append(word_tier)
                tg.append(ph_tier)

                if self.out_path is not None:
                    tg_path = self.out_path / "TextGrid" / wav_path.with_suffix(".TextGrid").name
                else:
                    tg_path = wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name
                tg_path.parent.mkdir(parents=True, exist_ok=True)
                tg.write(tg_path)
                
                print(f"成功处理并生成 TextGrid 文件: {tg_path}")

            except ValueError as e:
                print(f"错误: 文件 '{wav_path.name}' 包含无效的时间戳，已跳过整个文件的处理。")
                print(f"  - 详细错误信息: {e}")
                continue

    def save_confidence_fn(self):
        print("saving confidence...")
        folder_to_data = {}
        for wav_path, wav_length, words, confidence in self.predictions:
            folder = wav_path.parent
            if folder in folder_to_data:
                curr_data = folder_to_data[folder]
            else:
                curr_data = {
                    "name": [],
                    "confidence": [],
                }

            name = wav_path.with_suffix("").name
            curr_data["name"].append(name)
            curr_data["confidence"].append(confidence)
            folder_to_data[folder] = curr_data

        for folder, data in folder_to_data.items():
            df = pd.DataFrame(data)
            path = folder / "confidence"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "confidence.csv", index=False)

    def export(self, out_formats):
        self.save_textgrids()
        if "confidence" in out_formats:
            self.save_confidence_fn()
