import os
import pathlib

import pandas as pd
import yaml

language = "zh"
absolute = True
data_folder = "D:\python\DiffSinger\data\multi-data"
out_data_config = f"datasets_config_{language}.yaml"

csv_paths = []
for csv in pathlib.Path(data_folder).rglob("transcriptions.csv"):
    parent_dir = pathlib.Path(csv).parent
    print(parent_dir / f"{csv.name}")
    if os.path.exists(parent_dir / f"{csv.name}") and os.path.exists(parent_dir / "wavs"):
        with open(csv, "r", encoding="utf-8") as f:
            df = pd.read_csv(f)
            assert len(df) > 2, f"{csv.name} must have at least 2 rows."
            assert "ph_seq" in df.columns, f"{csv.name} not contains 'ph_seq' column."

            label_type = "full" if "ph_dur" in df.columns else "weak"

            csv_paths.append({
                "raw_data_dir": str(parent_dir.absolute()) if absolute else str(parent_dir),
                "label_type": label_type,
                "language": language,
                "test_prefixes": [df["name"][0]]
            })

with open(out_data_config, "w", encoding="utf-8") as f:
    yaml.dump({"datasets": csv_paths}, f)
