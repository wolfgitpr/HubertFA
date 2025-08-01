import random

import lightning as pl
import textgrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler_module

import networks.scheduler as scheduler_module
from evaluate import remove_ignored_phonemes, quantize_tier
from networks.layer.backbone.cvnt import CVNT
from networks.layer.backbone.unet import UNetBackbone
from networks.layer.block.resnet_block import ResidualBasicBlock
from networks.layer.scaling.stride_conv import DownSampling, UpSampling
from networks.loss.BinaryEMDLoss import BinaryEMDLoss
from networks.loss.GHMLoss import CTCGHMLoss, GHMLoss, MultiLabelGHMLoss
from networks.optimizer.muon import Muon_AdamW
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
        self.melspec_config: dict = melspec_config
        self.hubert_config: dict = hubert_config
        self.optimizer_config: dict = optimizer_config
        self.config: dict = config
        self.frame_length: float = melspec_config['hop_length'] / melspec_config['sample_rate']

        self.non_speech_target: list = self.vocab["non_speech_phonemes"]
        self.non_speech_mask_ratio: float = self.config["cvnt_arg"]["mask_ratio"]

        self.silent_phonemes: list = self.vocab["silent_phonemes"]
        self.ignored_phonemes: list = [x for x in self.silent_phonemes if x not in self.non_speech_target]
        self.language_prefix: bool = self.vocab.get("language_prefix", True)

        self.class_names: list = ['None', *self.non_speech_target]
        self.num_classes: int = len(self.class_names)

        assert self.num_classes > 1, "non_speech_phonemes must have at least one phoneme."

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

        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["vocab_size"] + 2
        )

        # cvnt
        self.cvnt = CVNT(config['cvnt_arg'], in_channels=self.hubert_config['channel'],
                         output_size=self.num_classes)

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_EMD_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "cvnt_loss",
            "total_loss",
        ]
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(
                        max_steps=optimizer_config["total_steps"] * 5
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
        self.decoder = AlignmentDecoder(self.vocab, self.class_names, self.melspec_config)

        # validation_step_outputs
        self.validation_step_outputs = {"losses": [], "tiers-2": [], "tiers-3": []}

        self.loss_weights = config.get('loss_weights', [1.0] * self.num_classes)
        self.focal_gamma = config.get('focal_gamma', 2.0)
        self.dice_smooth = config.get('dice_smooth', 1.0)

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
            self.unitsEncoder = UnitsEncoder(self.hubert_config, self.melspec_config, device=self.device)

    def make_non_speech_mask(self, shape, non_speech_intervals):
        B, T, C = shape
        non_speech_mask = torch.zeros((B, T, 1), dtype=torch.bool, device=self.device)
        if self.non_speech_mask_ratio > 0:
            mask_idxs = random.choices(range(B), k=int(B * self.non_speech_mask_ratio))
            for mask_idx in mask_idxs:
                non_speech_interval = non_speech_intervals[mask_idx]
                non_speech_idx = random.choices(range(len(non_speech_interval)),
                                                k=int(len(non_speech_interval) * self.non_speech_mask_ratio))
                for idx in non_speech_idx:
                    start_frame = non_speech_interval[idx][0]
                    end_frame = non_speech_interval[idx][1]
                    ph_len = end_frame - start_frame

                    mask_len = int(self.non_speech_mask_ratio * ph_len) + 1
                    offset = random.randint(0, ph_len - mask_len)
                    non_speech_mask[mask_idx, :, start_frame + offset:start_frame + offset + mask_len] = 1
        return non_speech_mask.bool()

    def predict_step(self, batch, batch_idx):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language, non_speech_phonemes = batch
        ph_seq = [f"{language}/{ph}" if ph not in self.silent_phonemes and self.language_prefix else ph for ph in
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
                cvnt_logits,  # [B,N,T]
            ) = self.forward(input_feature)

        words, confidence = self.decoder.decode(
            ph_frame_logits.float().cpu().numpy(),
            ph_edge_logits.float().cpu().numpy(),
            cvnt_logits.float().cpu().numpy(),
            wav_length, ph_seq, word_seq, ph_idx_to_word_idx,
            non_speech_phonemes=non_speech_phonemes
        )

        words.clear_language_prefix()
        return wav_path, wav_length, words, confidence

    def cross_entropy_and_focal_loss(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        weights = torch.tensor(self.loss_weights, device=logits.device)

        ce_loss = F.nll_loss(log_probs, targets, weight=weights)
        ce_per_element = F.nll_loss(log_probs, targets, reduction='none')
        pt = torch.exp(-ce_per_element)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_per_element).mean()

        return ce_loss, focal_loss

    def dice_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        batch_size, num_classes, seq_len = probs.shape

        target_mask = torch.zeros_like(probs)
        targets_expanded = targets.unsqueeze(1).expand(-1, num_classes, -1)
        target_mask.scatter_(1, targets_expanded, 1)

        target_mask = target_mask[:, 1:, :]
        pred_masks = probs[:, 1:, :]

        intersection = (pred_masks * target_mask).sum(dim=(0, 2))
        cardinality_pred = pred_masks.sum(dim=(0, 2))
        cardinality_target = target_mask.sum(dim=(0, 2))

        valid_classes = (cardinality_target > 0)
        if not valid_classes.any():
            return torch.tensor(0.0, device=logits.device)

        dice = (2 * intersection[valid_classes] + self.dice_smooth) / \
               (cardinality_pred[valid_classes] + cardinality_target[valid_classes] + self.dice_smooth)

        return (1 - dice).mean()

    def cvnt_loss(self, cvnt_logits, non_speech_targets):
        target_indices = torch.argmax(non_speech_targets, dim=1)
        ce_loss, focal_loss = self.cross_entropy_and_focal_loss(cvnt_logits, target_indices)
        dice_loss = self.dice_loss(cvnt_logits, target_indices)
        return 0.5 * ce_loss + 0.3 * focal_loss + 0.2 * dice_loss

    def _get_loss(
            self,
            ph_frame_logits,  # (B, T, vocab_size)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, T, vocab_size)
            cvnt_logits,  # [B, N, T]
            ph_frame_gt,  # (B, T)
            ph_edge_gt,  # (B, T)
            ph_seq_gt,  # (B, S)
            ph_seq_lengths_gt,  # (B)
            ph_mask,  # (B, vocab_size)
            input_feature_lengths,  # (B)
            label_type,  # (B)
            non_speech_target,
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

        cvnt_loss = ZERO

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

            cvnt_loss = self.cvnt_loss(cvnt_logits, non_speech_target)

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
            cvnt_loss
        ]

        return losses

    def forward(self,
                x,  # [B, T, C]
                ):
        cvnt_logits = self.cvnt(x)
        x = self.backbone(x)
        logits = self.head(x)  # [B, T, <vocab_size> + 2]
        ph_frame_logits = logits[:, :, 2:]  # [B, T, <vocab_size>]
        ph_edge_logits = logits[:, :, 0]  # [B, T]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)  # [B, T, <vocab_size>]
        return ph_frame_logits, ph_edge_logits, ctc_logits, cvnt_logits

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
                non_speech_target,
                non_speech_intervals,  # [B,N,T]
            ) = batch

            non_speech_mask = self.make_non_speech_mask(input_feature.shape, non_speech_intervals)
            masked_input = torch.where(
                non_speech_mask,
                torch.zeros_like(input_feature),
                input_feature
            )
            masked_non_speech_target = torch.where(
                non_speech_mask.transpose(1, 2),
                torch.zeros_like(non_speech_target),
                non_speech_target
            )

            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
                cvnt_logits,  # [B,N,T]
            ) = self.forward(masked_input)

            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                cvnt_logits,
                ph_frame,
                ph_edge,
                ph_id_seq,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                masked_non_speech_target,
                valid=False
            )

            schedule_weight = self._losses_schedulers_call()
            self._losses_schedulers_step()
            total_loss = (torch.stack(losses) * self.losses_weights * schedule_weight).sum()
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

            valid_mask = label_type >= 2  # [B]
            precision_metrics = {}

            if valid_mask.any():
                cvnt_logits_valid = cvnt_logits[valid_mask]  # [valid_B, N, T]
                non_speech_target_valid = masked_non_speech_target[valid_mask]  # [valid_B, N, T]

                with torch.no_grad():
                    probs = torch.softmax(cvnt_logits_valid, dim=1)  # [valid_B, N, T]
                    pred_classes = torch.argmax(probs, dim=1)  # [valid_B, T]

                    for class_idx, class_name in enumerate(self.class_names):
                        if class_idx == 0:
                            continue

                        pred_mask = (pred_classes == class_idx)  # [valid_B, T]
                        true_mask = non_speech_target_valid[:, class_idx, :] > 0.5  # [valid_B, T]

                        if true_mask.sum() > 0:
                            precision = (pred_mask & true_mask).sum() / pred_mask.sum().clamp_min(1e-8)
                            precision_metrics[f'non_speech_phonemes/{class_name}'] = precision

            for metric_name, value in precision_metrics.items():
                self.log(metric_name, value)

            return total_loss
        except Exception as e:
            print(f"Error: {e}. skip this batch.")
            return torch.tensor(torch.nan).to(self.device)

    def _get_evaluate_loss(self, tiers):
        metrics = {
            "BoundaryEditRatio": BoundaryEditRatio(),
            "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
            "VlabelerEditRatio_1-2frames": VlabelerEditRatio(move_min_frames=1, move_max_frames=2),
            "VlabelerEditRatio_3-5frames": VlabelerEditRatio(move_min_frames=3, move_max_frames=5),
            "VlabelerEditRatio_6-9frames": VlabelerEditRatio(move_min_frames=6, move_max_frames=9),
            "VlabelerEditRatio_10+frames": VlabelerEditRatio(move_min_frames=10, move_max_frames=10000)
        }

        if tiers:
            for pred_tier, target_tier in tiers:
                for metric in metrics.values():
                    pred_tier = remove_ignored_phonemes(pred_tier, self.ignored_phonemes)
                    target_tier = remove_ignored_phonemes(target_tier, self.ignored_phonemes)
                    metric.update(quantize_tier(pred_tier, self.frame_length),
                                  quantize_tier(target_tier, self.frame_length))

        result = {key: metric.compute() for key, metric in metrics.items()}

        vlabeler_loss = result["VlabelerEditRatio_1-2frames"] * 0.1 + result["VlabelerEditRatio_3-5frames"] * 0.2 + \
                        result["VlabelerEditRatio_6-9frames"] * 0.3 + result["VlabelerEditRatio_10+frames"] * 0.4
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
            non_speech_target,
            non_speech_intervals,
        ) = batch

        (
            ph_frame_logits,  # (B, T, vocab_size)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, T, vocab_size)
            cvnt_logits,  # [B,N,T]
        ) = self.forward(input_feature)

        ph_seq_g2p = []
        last_ph = ""
        for ph in ph_seq_raw[0]:
            temp_ph = ph if self.vocab["vocab"][ph] != 0 else "SP"
            if temp_ph == "SP" and last_ph == "SP":
                continue
            ph_seq_g2p.append(temp_ph)
            last_ph = temp_ph

        self.decoder.decode(
            ph_frame_logits.float().cpu().numpy(),
            ph_edge_logits.float().cpu().numpy(),
            cvnt_logits.float().cpu().numpy(),
            None, ph_seq_g2p, None, None, False,
            non_speech_phonemes=self.vocab["non_speech_phonemes"][1:]
        )

        if dataloader_idx == 0 or self.config.get("get_evaluate_loss", False):
            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                cvnt_logits,
                ph_frame,
                ph_edge,
                ph_seq_id,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                non_speech_target,
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

            for mark, time in zip(self.decoder.ph_seq_pred, self.decoder.ph_intervals_pred):
                pred_tier.addPoint(textgrid.Point(float(time[0]), mark))
            self.validation_step_outputs[f"tiers-{label_type_id}"].append((pred_tier, target_tier))

        if ((dataloader_idx == 0 or self.config.get("draw_evaluate", False))
                and batch_idx < self.config.get("num_valid_plots", 20)):
            melspec = melspec[0].cpu().numpy().squeeze()  # [N_Mel,T]
            fig = self.decoder.plot(melspec, ph_time_gt=ph_time_raw[0])
            self.logger.experiment.add_figure(f"valid/plot_{name[0]}", fig, self.global_step)

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
        optimizer = Muon_AdamW(
            self,
            lr=self.optimizer_config["lr"],
            muon_args=self.optimizer_config["muon_args"],
            adamw_args=self.optimizer_config["adamw_args"],
            weight_decay=self.optimizer_config["muon_args"]["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.ExponentialLR(
                optimizer,
                gamma=self.optimizer_config["gamma"]
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
