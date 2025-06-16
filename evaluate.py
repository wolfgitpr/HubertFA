import json
import pathlib
import warnings
from typing import Dict

import click
import tqdm
from textgrid import PointTier, Point

from tools import label
from tools.metrics import (
    CustomPointTier,
    BoundaryEditRatio,
    BoundaryEditRatioWeighted,
    IntersectionOverUnion,
    Metric,
    VlabelerEditRatio,
)


def remove_ignored_phonemes(ignored_phonemes_list: list[str], point_tier: PointTier):
    res_tier = CustomPointTier(name=point_tier.name)
    if point_tier[0].mark not in ignored_phonemes_list:
        res_tier.addPoint(point_tier[0])
    for i in range(len(point_tier) - 1):
        if point_tier[i].mark in ignored_phonemes_list and point_tier[i + 1].mark in ignored_phonemes_list:
            continue
        res_tier.addPoint(point_tier[i + 1])

    return res_tier


def quantize_tier(tier: PointTier, frame_length: float) -> CustomPointTier:
    """Quantize tier times to frame boundaries"""
    new_tier = CustomPointTier(name=tier.name)
    points = sorted(tier.points, key=lambda p: p.time)  # 确保时间点有序
    for point in points:
        # Quantize to nearest frame boundary
        quantized_time = round(point.time / frame_length)
        new_tier.addPoint(Point(quantized_time, point.mark))
    return new_tier


@click.command(
    help="Calculate metrics between the FA predictions and the targets (ground truth)."
)
@click.argument(
    "pred",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="PRED_DIR",
)
@click.argument(
    "target",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="TARGET_DIR",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Compare files in subdirectories recursively",
)
@click.option(
    "--strict", "-s", is_flag=True, help="Raise errors on mismatching phone sequences"
)
@click.option(
    "--ignore",
    type=str,
    default="",  # AP,SP,<AP>,<SP>,,pau,cl
    help="Ignored phone marks, split by commas",
    show_default=True,
)
@click.option(
    "--frame_length",
    "-fl",
    type=float,
    default=512 / 44100,
    help="Frame length in seconds for quantization (default: 512/44100)",
    show_default=True,
)
def main(pred: str, target: str, recursive: bool, strict: bool, ignore: str, frame_length: float):
    pred_dir = pathlib.Path(pred)
    target_dir = pathlib.Path(target)
    if recursive:
        iterable = list(pred_dir.rglob("*.TextGrid"))
    else:
        iterable = list(pred_dir.glob("*.TextGrid"))
    ignored = [ph.strip() for ph in ignore.split(",") if ph.strip()]
    metrics: Dict[str, Metric] = {
        "BoundaryEditRatio": BoundaryEditRatio(),
        "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
        "VlabelerEditRatio_1-2frames": VlabelerEditRatio(move_min_frames=1, move_max_frames=2),
        "VlabelerEditRatio_3-5frames": VlabelerEditRatio(move_min_frames=3, move_max_frames=5),
        "VlabelerEditRatio_6-9frames": VlabelerEditRatio(move_min_frames=6, move_max_frames=9),
        "VlabelerEditRatio_10+frames": VlabelerEditRatio(move_min_frames=10, move_max_frames=10000),
        "IntersectionOverUnion": IntersectionOverUnion(),
    }

    cnt = 0
    for pred_file in tqdm.tqdm(iterable):
        target_file = target_dir / pred_file.relative_to(pred_dir)
        if not target_file.exists():
            warnings.warn(
                f'The prediction file "{pred_file}" has no matching target file, '
                f'which should be "{target_file}".',
                category=UserWarning,
            )
            continue

        pred_tier = label.textgrid_from_file(pred_file)[-1]
        target_tier = label.textgrid_from_file(target_file)[-1]

        # Remove ignored phonemes
        pred_tier = remove_ignored_phonemes(ignored, pred_tier)
        target_tier = remove_ignored_phonemes(ignored, target_tier)

        # Quantize to frame boundaries
        pred_tier = quantize_tier(pred_tier, frame_length)
        target_tier = quantize_tier(target_tier, frame_length)

        for metric in metrics.values():
            try:
                metric.update(pred_tier, target_tier)
            except AssertionError as e:
                if not strict:
                    warnings.warn(
                        f"Failed to evaluate metric {metric.__class__.__name__} for file {pred_file}: {e}",
                        category=UserWarning,
                    )
                    continue
                else:
                    raise e

        cnt += 1

    if cnt == 0:
        raise RuntimeError(
            "Unable to compare any files in the given directories. "
            "Matching files should have same names and same relative paths, "
            "containing the same phone sequences except for spaces."
        )
    result = {key: metric.compute() for key, metric in metrics.items()}
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
