from pathlib import Path
from typing import Union

import textgrid as tg


def interval_tier_to_point_tier(tier: tg.IntervalTier) -> tg.PointTier:
    point_tier = tg.PointTier(name=tier.name)
    point_tier.add(0.0, "")
    for interval in tier:
        if point_tier[-1].mark == "" and point_tier[-1].time == interval.minTime:
            point_tier[-1].mark = interval.mark
        else:
            point_tier.add(interval.minTime, interval.mark)
        point_tier.add(interval.maxTime, "")

    return point_tier


def point_tier_to_interval_tier(tier: tg.PointTier) -> tg.IntervalTier:
    interval_tier = tg.IntervalTier(name=tier.name)
    for idx in range(len(tier) - 1):
        interval_tier.add(tier[idx].time, tier[idx + 1].time, tier[idx].mark)
    return interval_tier


def textgrid_from_file(textgrid_path: Union[str, Path]) -> tg.TextGrid:
    """Read a TextGrid file and return a TextGrid object."""
    _textgrid = tg.TextGrid()
    _textgrid.read(textgrid_path, encoding="utf-8")
    for idx, tier in enumerate(_textgrid):
        if isinstance(tier, tg.IntervalTier):
            _textgrid.tiers[idx] = interval_tier_to_point_tier(tier)

    return _textgrid


def save_textgrid(path: str, _textgrid: tg.TextGrid) -> None:
    """Save a TextGrid object to a TextGrid file."""
    for i in range(len(_textgrid)):
        if _textgrid[i].maxTime is None:
            _textgrid[i].maxTime = _textgrid[i][-1].time
        if isinstance(_textgrid[i], tg.PointTier):
            _textgrid.tiers[i] = point_tier_to_interval_tier(_textgrid[i])
    _textgrid.write(path)


if __name__ == "__main__":
    textgrid = textgrid_from_file("test/label/tg.TextGrid")
    save_textgrid("test/label/tg_out.TextGrid", textgrid)
