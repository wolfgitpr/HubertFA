import warnings

import lightning as pl
import textgrid
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module

import networks.scheduler as scheduler_module
from evaluate import remove_ignored_phonemes
from networks.layer.backbone.unet import UNetBackbone
from networks.layer.block.conformer import ConformerBlock
from networks.layer.block.resnet_block import ResidualBasicBlock
from networks.layer.scaling.stride_conv import DownSampling, UpSampling
from networks.loss.BinaryEMDLoss import BinaryEMDLoss
from networks.loss.GHMLoss import CTCGHMLoss, GHMLoss, MultiLabelGHMLoss
from tools.alignment_decoder import AlignmentDecoder
from tools.encoder import UnitsEncoder
from tools.load_wav import load_wav
from tools.metrics import BoundaryEditRatio, BoundaryEditRatioWeighted, VlabelerEditRatio, CustomPointTier


class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
            self,
            vocab: dict,
            model_config: dict,
            hubert_config: dict,
            melspec_config: dict,
            optimizer_config: dict,
            loss_config: dict,
            config: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab: dict = vocab
        self.silent_phonemes: list = self.vocab["silent_phonemes"]
        self.global_phonemes: list = self.vocab["global_phonemes"]
        self.ignored_phones: list = self.silent_phonemes + self.global_phonemes
        self.language_prefix = self.vocab.get("language_prefix", True)

        self.backbone_type = model_config["backbone"]

        if self.backbone_type == 'unet':
            self.backbone = UNetBackbone(
                input_dims=hubert_config["channel"],
                output_dims=model_config["hidden_dims"],
                hidden_dims=model_config["hidden_dims"],
                block=ResidualBasicBlock,
                down_sampling=DownSampling,
                up_sampling=UpSampling,
                down_sampling_factor=model_config["down_sampling_factor"],  # 3
                down_sampling_times=model_config["down_sampling_times"],  # 7
                channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
            )
        elif self.backbone_type == 'conformer':
            self.input_proj = nn.Linear(hubert_config["channel"], model_config["hidden_dims"])
            self.backbone = nn.ModuleList([
                ConformerBlock(
                    dim=model_config["hidden_dims"],
                    heads=config["model"].get("conformer_heads", 4),
                    ff_expansion=config["model"].get("conformer_ff_expansion", 4),
                    conv_kernel=config["model"].get("conformer_kernel_size", 31),
                    dropout=config["model"].get("conformer_dropout", 0.1)
                )
                for _ in range(config["model"].get("num_conformer_layers", 2))
            ])
        else:
            warnings.warn(f"{model_config['backbone']} is not supported.")

        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["vocab_size"] + 2
        )
        self.melspec_config = melspec_config
        self.hubert_config = hubert_config
        self.optimizer_config = optimizer_config
        self.config = config

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_EMD_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "total_loss",
        ]
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(
                        max_steps=optimizer_config["total_steps"]
                    )
                )
            else:
                self.losses_schedulers.append(scheduler_module.NoneScheduler())

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(
            self.vocab["vocab_size"],
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )
        self.ph_edge_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.EMD_loss_fn = BinaryEMDLoss()
        self.ph_edge_diff_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=1 - 1e-3)

        self.unitsEncoder = None

        self.decoder = AlignmentDecoder(self.vocab, self.melspec_config)

        # validation_step_outputs
        self.validation_step_outputs = {"losses": [], "tiers-2": [], "tiers-3": []}

    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["vocab_size"] == pretrained_model.vocab["vocab_size"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["vocab_size"] + 2
            )

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        # resume loss schedulers
        for scheduler in self.losses_schedulers:
            scheduler.resume(self.global_step)
        self.losses_weights = self.losses_weights.to(self.device)

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(self.device)

    def on_predict_start(self):
        if self.unitsEncoder is None:
            self.unitsEncoder = UnitsEncoder(self.hubert_config, self.melspec_config, self.device)

    def predict_step(self, batch, batch_idx):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language = batch
        ph_seq = [f"{language}/{ph}" if ph not in self.ignored_phones and self.language_prefix else ph for ph in
                  ph_seq]
        waveform = load_wav(wav_path, self.device, self.melspec_config["sample_rate"])
        wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
        input_feature = self.unitsEncoder.forward(waveform.unsqueeze(0), self.melspec_config["sample_rate"],
                                                  self.melspec_config["hop_length"])  # [B, T, C]

        with torch.no_grad():
            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(input_feature)

        (
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
            confidence
        ) = self.decoder.decode(
            ph_frame_logits, ph_edge_logits, ctc_logits, wav_length, ph_seq, word_seq, ph_idx_to_word_idx
        )

        ph_seq = [x.split("/")[-1] for x in ph_seq]

        return (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        )

    def _get_loss(
            self,
            ph_frame_logits,  # (B, T, vocab_size)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, T, vocab_size)
            ph_frame_gt,  # (B, T)
            ph_edge_gt,  # (B, T)
            ph_seq_gt,  # (B, S)
            ph_seq_lengths_gt,  # (B)
            ph_mask,  # (B, vocab_size)
            input_feature_lengths,  # (B)
            label_type,  # (B)
            valid=False
    ):
        device = ph_frame_logits.device
        ZERO = torch.tensor(0.0, device=device, requires_grad=True)

        full_mask = label_type >= 2
        weak_mask = label_type >= 1

        time_mask = torch.arange(ph_frame_logits.shape[1], device=device)[None, :] < input_feature_lengths[:, None]
        time_mask = time_mask.float()

        ph_frame_GHM_loss = ZERO
        ph_edge_GHM_loss = ZERO
        ph_edge_EMD_loss = ZERO
        ph_edge_diff_loss = ZERO

        if torch.any(full_mask):
            selected_logits = ph_frame_logits[full_mask]
            selected_edges = ph_edge_logits[full_mask]
            selected_gt = ph_frame_gt[full_mask]
            selected_edge_gt = ph_edge_gt[full_mask]
            selected_ph_mask = ph_mask[full_mask]
            selected_time_mask = time_mask[full_mask]

            edge_diff_gt = (selected_edge_gt[:, 1:] - selected_edge_gt[:, :-1])
            edge_diff_gt = (edge_diff_gt + 1) / 2

            edge_diff_pred = torch.sigmoid(selected_edges[:, 1:]) - torch.sigmoid(selected_edges[:, :-1])
            edge_diff_pred = (edge_diff_pred + 1) / 2

            valid_diff_mask = selected_time_mask[:, 1:] > 0
            ph_edge_diff_loss = self.ph_edge_diff_GHM_loss_fn(
                edge_diff_pred.unsqueeze(-1),  # (B,T-1,1)
                edge_diff_gt.unsqueeze(-1),  # (B,T-1,1)
                valid_diff_mask.unsqueeze(-1),
                valid
            ) if valid_diff_mask.any() else ZERO

            ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
                selected_logits, selected_gt,
                selected_ph_mask.unsqueeze(1) * selected_time_mask.unsqueeze(-1),
                valid
            )

            ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(
                selected_edges.unsqueeze(-1),
                selected_edge_gt.unsqueeze(-1),
                selected_time_mask.unsqueeze(-1),
                valid
            )

            ph_edge_EMD_loss = self.EMD_loss_fn(
                torch.sigmoid(selected_edges) * selected_time_mask,
                selected_edge_gt * selected_time_mask
            )

        ctc_GHM_loss = ZERO
        if torch.any(weak_mask):
            weak_logits = ctc_logits[weak_mask]
            weak_seq_gt = ph_seq_gt[weak_mask]
            weak_seq_len = ph_seq_lengths_gt[weak_mask]
            weak_time_mask = input_feature_lengths[weak_mask]

            ctc_log_probs = torch.log_softmax(weak_logits, dim=-1)
            ctc_log_probs = ctc_log_probs.permute(1, 0, 2)  # (T, B, C)

            ctc_GHM_loss = self.CTC_GHM_loss_fn(
                ctc_log_probs,
                weak_seq_gt,
                weak_time_mask,
                weak_seq_len,
                valid
            )

        losses = [
            ph_frame_GHM_loss,
            ph_edge_GHM_loss,
            ph_edge_EMD_loss,
            ph_edge_diff_loss,
            ctc_GHM_loss,
        ]

        return losses

    def forward(self,
                x,  # [B, T, C]
                ):
        if self.backbone_type == 'unet':
            x = self.backbone(x)
        if self.backbone_type == 'conformer':
            x = self.input_proj(x)
            for layer in self.backbone:
                x = layer(x)

        logits = self.head(x)  # [B, T, <vocab_size> + 2]
        ph_frame_logits = logits[:, :, 2:]  # [B, T, <vocab_size>]
        ph_edge_logits = logits[:, :, 0]  # [B, T]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)  # [B, T, <vocab_size>]
        return ph_frame_logits, ph_edge_logits, ctc_logits

    def training_step(self, batch, batch_idx):
        try:
            (
                input_feature,  # (B, n_mels, T)
                input_feature_lengths,  # (B)
                ph_seq,  # (B S)
                ph_id_seq,  # (B S)
                ph_seq_lengths,  # (B)
                ph_edge,  # (B, T)
                ph_frame,  # (B, T)
                ph_mask,  # (B vocab_size)
                label_type,  # (B)
                melspec,
                ph_time,
                name,
                ph_seq_raw,
                ph_time_raw,
            ) = batch

            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(input_feature)

            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                ph_frame,
                ph_edge,
                ph_id_seq,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                valid=False
            )

            schedule_weight = self._losses_schedulers_call()
            self._losses_schedulers_step()
            total_loss = (
                    torch.stack(losses) * self.losses_weights * schedule_weight
            ).sum()
            losses.append(total_loss)

            log_dict = {
                f"train_loss/{k}": v
                for k, v in zip(self.losses_names, losses)
                if v != 0
            }
            log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_dict.update(
                {
                    f"scheduler/{k}": v
                    for k, v in zip(self.losses_names, schedule_weight)
                    if v != 1
                }
            )
            self.log_dict(log_dict)
            return total_loss
        except Exception as e:
            print(f"Error: {e}. skip this batch.")
            return torch.tensor(torch.nan).to(self.device)

    def _get_evaluate_loss(self, tiers):
        metrics = {
            "BoundaryEditRatio": BoundaryEditRatio(),
            "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
            "VlabelerEditRatio10-20ms": VlabelerEditRatio(move_min=0.01, move_max=0.02),
            "VlabelerEditRatio20-50ms": VlabelerEditRatio(move_min=0.02, move_max=0.05),
            "VlabelerEditRatio50-100ms": VlabelerEditRatio(move_min=0.05, move_max=0.1),
            "VlabelerEditRatio100-5000ms": VlabelerEditRatio(move_min=0.1, move_max=5.0)
        }

        if tiers:
            for pred_tier, target_tier in tiers:
                for metric in metrics.values():
                    pred_tier = remove_ignored_phonemes(self.ignored_phones, pred_tier)
                    target_tier = remove_ignored_phonemes(self.ignored_phones, target_tier)
                    metric.update(pred_tier, target_tier)

        result = {key: metric.compute() for key, metric in metrics.items()}

        vlabeler_loss = result["VlabelerEditRatio10-20ms"] * 0.1 + result["VlabelerEditRatio20-50ms"] * 0.2 + \
                        result["VlabelerEditRatio50-100ms"] * 0.3 + result["VlabelerEditRatio100-5000ms"] * 0.4
        result["vlabeler_loss"] = vlabeler_loss
        result["total"] = vlabeler_loss * 0.5 + result["BoundaryEditRatioWeighted"] * 0.5
        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (
            input_feature,  # (B, n_mels, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_id,  # (B S)
            ph_seq_lengths,  # (B)
            ph_edge,  # (B, T)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            label_type,  # (B)
            melspec,
            ph_time,
            name,
            ph_seq_raw,
            ph_time_raw,
        ) = batch

        (
            ph_frame_logits,  # (B, T, vocab_size)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, T, vocab_size)
        ) = self.forward(input_feature)

        ph_seq_g2p = []
        last_ph = ""
        for ph in ph_seq_raw[0]:
            temp_ph = ph if self.vocab["vocab"][ph] != 0 else "SP"
            if temp_ph == "SP" and last_ph == "SP":
                continue
            ph_seq_g2p.append(temp_ph)
            last_ph = temp_ph

        (
            ph_seq_pred, ph_intervals_pred, _, _, _
        ) = self.decoder.decode(
            ph_frame_logits, ph_edge_logits, ctc_logits, None, ph_seq_g2p, None, None, False
        )

        if ((dataloader_idx == 0 or self.config.get("draw_evaluate", False))
                and batch_idx < self.config.get("num_valid_plots", 20)):
            fig = self.decoder.plot(melspec, ph_time_raw[0])
            self.logger.experiment.add_figure(f"valid/plot_{name[0]}", fig, self.global_step)

        if dataloader_idx == 0 or self.config.get("get_evaluate_loss", False):
            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                ph_frame,
                ph_edge,
                ph_seq_id,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                valid=True
            )

            weights = self._losses_schedulers_call() * self.losses_weights
            total_loss = (torch.stack(losses) * weights).sum()
            losses.append(total_loss)
            losses = torch.stack(losses)

            self.validation_step_outputs["losses"].append(losses)

        label_type_id = label_type.cpu().numpy()[0]
        if label_type_id >= 2:
            pred_tier = CustomPointTier(name="phones")
            target_tier = CustomPointTier(name="phones")

            for mark, time in zip(ph_seq[0], ph_time[0].cpu().numpy()):
                target_tier.addPoint(textgrid.Point(float(time), mark))

            for mark, time in zip(ph_seq_pred, ph_intervals_pred):
                pred_tier.addPoint(textgrid.Point(float(time[0]), mark))
            self.validation_step_outputs[f"tiers-{label_type_id}"].append((pred_tier, target_tier))

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs["losses"], dim=0)
        losses = (losses / ((losses > 0).sum(dim=0, keepdim=True) + 1e-6)).sum(dim=0)
        self.log_dict(
            {f"valid/{k}": v for k, v in zip(self.losses_names, losses) if v != 0}
        )

        val_loss = self._get_evaluate_loss(self.validation_step_outputs.get(f"tiers-2", []))
        for metric_name, metric_value in val_loss.items():
            self.log_dict({f"valid_evaluate/{metric_name}": metric_value})
        self.validation_step_outputs[f"tiers-2"].clear()

        evaluate_loss = self._get_evaluate_loss(self.validation_step_outputs.get(f"tiers-3", []))
        for metric_name, metric_value in evaluate_loss.items():
            self.log_dict({f"unseen_evaluate/{metric_name}": metric_value})
        self.validation_step_outputs[f"tiers-3"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone.parameters(),
                    "lr": self.optimizer_config["lr"]["backbone"],
                },
                {
                    "params": self.head.parameters(),
                    "lr": self.optimizer_config["lr"]["head"],
                },
            ],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.ExponentialLR(
                optimizer,
                gamma=self.optimizer_config["gamma"]
            ),
            "interval": "step",
        }

        for k, v in self.optimizer_config["freeze"].items():
            if v:
                getattr(self, k).requires_grad_(False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
