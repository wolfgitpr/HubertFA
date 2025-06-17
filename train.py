import os
import pathlib
import shutil

import click
import lightning as pl
import torch
from torch.utils.data import DataLoader

from networks.task.forced_alignment import LitForcedAlignmentTask
from tools.config_utils import load_yaml
from tools.dataset import MixedDataset, BinningAudioBatchSampler, collate_fn
from tools.train_callbacks import StepProgressBar, RecentCheckpointsCallback, MonitorCheckpointsCallback


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="configs/train_config.yaml",
    show_default=True,
    help="training config path",
)
@click.option(
    "--pretrained_model_path",
    "-p",
    type=str,
    default=None,
    show_default=True,
    help="pretrained model path. if None, training from scratch",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    show_default=True,
    help="resume training from checkpoint",
)
def main(config: str, pretrained_model_path, resume):
    os.environ[
        "TORCH_CUDNN_V8_API_ENABLED"
    ] = "1"  # Prevent unacceptable slowdowns when using 16 precision

    config = load_yaml(config)

    binary_folder = pathlib.Path(config["binary_folder"])
    vocab = load_yaml(binary_folder / "vocab.yaml")

    config_global = load_yaml(binary_folder / "config.yaml")
    config.update(config_global)

    save_model_folder = pathlib.Path("ckpt") / config["model_name"]

    os.makedirs(save_model_folder, exist_ok=True)

    shutil.copy(binary_folder / "vocab.yaml", save_model_folder)
    shutil.copy(binary_folder / "config.yaml", save_model_folder)

    for lang, dict_path in vocab["dictionaries"].items():
        shutil.copy(binary_folder / dict_path, save_model_folder)
        print(f'| Copied dictionary for language-{lang}-{dict_path} to {save_model_folder}.')

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    pl.seed_everything(config["random_seed"], workers=True)

    # define dataset
    num_workers = config['dataloader_workers']
    train_dataset = MixedDataset(binary_folder, prefix="train")
    train_sampler = BinningAudioBatchSampler(
        train_dataset.get_wav_lengths(),
        config["batch_max_length"],
        config["binning_length"],
        config["drop_last"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    valid_dataset = MixedDataset(binary_folder, prefix="valid")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    evaluate_dataset = MixedDataset(binary_folder, prefix="evaluate")
    evaluate_dataloader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # model
    lightning_alignment_model = LitForcedAlignmentTask(
        vocab,
        config["model"],
        config["hubert_config"],
        config["melspec_config"],
        config["optimizer_config"],
        config["loss_config"],
        config
    )

    recent_checkpoints_callback = RecentCheckpointsCallback(
        dirpath=save_model_folder,
        save_top_k=config["save_top_k"],
        save_every_steps=config["save_every_steps"],
    )

    stepProgressBar = StepProgressBar()

    evaluate_checkpoint = MonitorCheckpointsCallback(
        dirpath=save_model_folder,
        monitor="unseen_evaluate/total",
        mode="min",
        save_top_k=5,
    )

    # trainer
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        default_root_dir=save_model_folder,
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        max_epochs=-1,
        max_steps=config["optimizer_config"]["total_steps"],
        callbacks=[recent_checkpoints_callback, evaluate_checkpoint, stepProgressBar],
    )

    ckpt_path = None
    if pretrained_model_path is not None:
        # use pretrained model TODO: load pretrained model
        pretrained = LitForcedAlignmentTask.load_from_checkpoint(pretrained_model_path)
        lightning_alignment_model.load_pretrained(pretrained)
    elif resume:
        # resume training state
        ckpt_path_list = save_model_folder.glob("*.ckpt")
        ckpt_path_list = sorted(
            ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True
        )
        ckpt_path = str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None

    # start training
    trainer.fit(
        model=lightning_alignment_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[valid_dataloader, evaluate_dataloader],
        ckpt_path=ckpt_path,
    )

    # Discard the optimizer and save
    trainer.save_checkpoint(
        str(pathlib.Path("ckpt") / config["model_name"]) + ".ckpt", weights_only=True
    )


if __name__ == "__main__":
    main()
