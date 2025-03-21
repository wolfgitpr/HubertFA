import os
import pathlib

import click
import lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from networks.utils.dataset import MixedDataset, WeightedBinningAudioBatchSampler, collate_fn
from networks.task.forced_alignment import LitForcedAlignmentTask

from lightning.pytorch.callbacks import Callback


class RecentCheckpointsCallback(Callback):
    def __init__(self, dirpath, save_top_k=5, save_every_steps=5000, filename="checkpoint-{step}"):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self.filename = filename
        self.saved_checkpoints = []
        self.save_every_steps = save_every_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_every_steps == 0:
            checkpoint_path = os.path.join(
                self.dirpath,
                self.filename.format(step=trainer.global_step) + ".ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)

            if len(self.saved_checkpoints) > self.save_top_k:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)


@click.command()
@click.option(
    "--config_path",
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
def main(config_path: str, pretrained_model_path, resume):
    os.environ[
        "TORCH_CUDNN_V8_API_ENABLED"
    ] = "1"  # Prevent unacceptable slowdowns when using 16 precision

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(pathlib.Path(config["binary_folder"]) / "vocab.yaml") as f:
        vocab = yaml.safe_load(f)
    vocab_text = yaml.safe_dump(vocab)

    with open(pathlib.Path(config["binary_folder"]) / "vowel.yaml") as f:
        vowel = yaml.safe_load(f)
    vowel_text = yaml.safe_dump(vowel)

    with open(pathlib.Path(config["binary_folder"]) / "global_config.yaml") as f:
        config_global = yaml.safe_load(f)
    config.update(config_global)

    config["data_augmentation_size"] = 0

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    pl.seed_everything(config["random_seed"], workers=True)

    # define dataset
    num_workers = config['dataloader_workers']
    train_dataset = MixedDataset(
        config["data_augmentation_size"], config["binary_folder"], prefix="train"
    )
    train_sampler = WeightedBinningAudioBatchSampler(
        train_dataset.get_label_types(),
        train_dataset.get_wav_lengths(),
        config["oversampling_weights"],
        config["batch_max_length"] / (2 if config["data_augmentation_size"] > 0 else 1),
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

    valid_dataset = MixedDataset(0, config["binary_folder"], prefix="valid")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # model
    lightning_alignment_model = LitForcedAlignmentTask(
        vocab_text,
        vowel_text,
        config["model"],
        config["hubert_config"],
        config["melspec_config"],
        config["optimizer_config"],
        config["loss_config"]
    )

    recent_checkpoints_callback = RecentCheckpointsCallback(
        dirpath=str(pathlib.Path("ckpt") / config["model_name"]),
        save_top_k=config["save_top_k"],
        save_every_steps=config["save_every_steps"],
        filename="checkpoint-{step}",
    )

    # trainer
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        default_root_dir=str(pathlib.Path("ckpt") / config["model_name"]),
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        max_epochs=-1,
        max_steps=config["optimizer_config"]["total_steps"],
        callbacks=[recent_checkpoints_callback],
    )

    ckpt_path = None
    if pretrained_model_path is not None:
        # use pretrained model TODO: load pretrained model
        pretrained = LitForcedAlignmentTask.load_from_checkpoint(pretrained_model_path)
        lightning_alignment_model.load_pretrained(pretrained)
    elif resume:
        # resume training state
        ckpt_path_list = (pathlib.Path("ckpt") / config["model_name"]).rglob("*.ckpt")
        ckpt_path_list = sorted(
            ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True
        )
        ckpt_path = str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None

    # start training
    trainer.fit(
        model=lightning_alignment_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=ckpt_path,
    )

    # Discard the optimizer and save
    trainer.save_checkpoint(
        str(pathlib.Path("ckpt") / config["model_name"]) + ".ckpt", weights_only=True
    )


if __name__ == "__main__":
    main()
