import torch
import torch.nn as nn


def get_bin_index(tensor, num_bins):
    return (tensor * num_bins).floor().long().clamp(0, num_bins - 1)


def update_ema(ema, alpha, num_bins, hist):
    hist = hist / hist.sum().clamp(min=1e-10) * num_bins
    ema = ema * alpha + (1 - alpha) * hist
    return ema / ema.sum().clamp(min=1e-10) * num_bins


class CTCGHMLoss(nn.Module):
    def __init__(self, num_bins=10, alpha=0.999):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(reduction="none")
        self.num_bins = num_bins
        self.register_buffer("ema", torch.ones(num_bins))
        self.alpha = alpha

    def forward(self, log_probs, targets, input_lengths, target_lengths, valid=False):
        if log_probs.size(0) == 0:
            return torch.tensor(0.0, device=log_probs.device)

        device = log_probs.device
        try:
            raw_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        except RuntimeError:
            raw_loss = self.ctc_loss_fn(
                log_probs.cpu(), targets.cpu(),
                input_lengths.cpu(), target_lengths.cpu()
            ).to(device)

        loss_for_ema = (-raw_loss / input_lengths).exp().clamp(1e-6, 1 - 1e-6)
        bin_indices = get_bin_index(loss_for_ema, self.num_bins)
        weights = self.ema[bin_indices].detach()
        loss_weighted = raw_loss / (weights + 1e-10)

        if not valid:
            hist = torch.histc(loss_for_ema, self.num_bins, min=0, max=1)
            self.ema.data = update_ema(self.ema, self.alpha, self.num_bins, hist)

        return loss_weighted.mean()


class BCEGHMLoss(nn.Module):
    def __init__(self, num_bins=10, alpha=0.999, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")
        self.num_bins = num_bins
        self.register_buffer("GD_stat_ema", torch.ones(num_bins))
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, pred_prob, target_prob, mask=None, valid=False):
        if pred_prob.size(0) == 0:
            return torch.tensor(0.0, device=pred_prob.device)

        mask = mask.expand_as(pred_prob) if mask is not None else 1.0
        target_prob = target_prob.clamp(self.label_smoothing, 1 - self.label_smoothing)

        raw_loss = self.loss_fn(pred_prob, target_prob)
        grad_mag = (pred_prob - target_prob).abs()
        bin_indices = get_bin_index(grad_mag, self.num_bins)
        weights = 1 / self.GD_stat_ema[bin_indices].detach()

        loss_weighted = (raw_loss * weights * mask).sum() / mask.sum().clamp(min=1e-10)

        if not valid:
            hist = torch.bincount(bin_indices.flatten(),
                                  weights=mask.flatten().float(),
                                  minlength=self.num_bins)
            self.GD_stat_ema.data = update_ema(self.GD_stat_ema, self.alpha, self.num_bins, hist)

        return loss_weighted


class MultiLabelGHMLoss(nn.Module):
    def __init__(self, num_classes, num_bins=10, alpha=0.999, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.num_bins = num_bins
        self.register_buffer("GD_stat_ema", torch.ones(num_bins))
        self.register_buffer("label_stat_ema_each_class", torch.ones(num_classes * 3))
        self.num_classes = num_classes
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits, target_prob, mask=None, valid=False):
        if pred_logits.size(0) == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        mask = mask.expand_as(pred_logits) if mask is not None else 1.0
        target_prob = target_prob.clamp(self.label_smoothing, 1 - self.label_smoothing)

        raw_loss = self.loss_fn(pred_logits, target_prob)
        pred_prob = torch.sigmoid(pred_logits)
        grad_mag = (pred_prob - target_prob).abs()
        bin_indices = get_bin_index(grad_mag, self.num_bins)
        GD_weights = 1 / self.GD_stat_ema[bin_indices].detach()

        target_porb_index = (target_prob * 3).floor().long().clamp(0, 2) + 3 * torch.arange(
            self.num_classes, device=target_prob.device).unsqueeze(0)
        classes_weights = 1 / self.label_stat_ema_each_class[target_porb_index].detach()
        weights = torch.sqrt(GD_weights * classes_weights)

        loss_weighted = (raw_loss * weights * mask).sum() / mask.sum().clamp(min=1e-10)

        if not valid:
            GD_hist = torch.bincount(bin_indices.flatten(),
                                     weights=mask.flatten().float(),
                                     minlength=self.num_bins)
            label_hist = torch.bincount(target_porb_index.flatten(),
                                        weights=mask.flatten().float(),
                                        minlength=self.num_classes * 3)
            self.GD_stat_ema.data = update_ema(self.GD_stat_ema, self.alpha, self.num_bins, GD_hist)
            self.label_stat_ema_each_class.data = update_ema(
                self.label_stat_ema_each_class, self.alpha, self.num_classes * 3, label_hist)

        return loss_weighted


class GHMLoss(torch.nn.Module):
    def __init__(self, num_classes, num_bins=10, alpha=1 - 1e-6, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_ema", torch.ones(num_classes))
        self.num_bins = num_bins
        self.register_buffer("GD_ema", torch.ones(num_bins))
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits, target_label, mask=None, valid=False):
        if len(pred_logits) <= 0:
            return torch.tensor(0.0).to(pred_logits.device)

        # pred: [B, T, C]
        assert len(pred_logits.shape) == 3 and pred_logits.shape[-1] == self.num_classes

        # target: [B, T]
        assert len(target_label.shape) == 2
        assert target_label.shape[0] == pred_logits.shape[0]
        assert target_label.shape[1] == pred_logits.shape[1]
        target_label = target_label.long()

        # mask: [B, T] or [B, T, C]
        if mask is None:
            mask = torch.ones_like(pred_logits).to(pred_logits.device)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
        assert mask.shape[0] == target_label.shape[0]
        assert mask.shape[1] == target_label.shape[1]
        assert mask.shape[-1] == 1 or mask.shape[-1] == self.num_classes
        time_mask = mask.any(dim=-1)  # [B, T]

        pred_logits = pred_logits - 1e9 * mask.logical_not().float()
        target_prob = (
            nn.functional.one_hot(target_label, num_classes=self.num_classes)
            .float()
            .clamp(self.label_smoothing, 1 - self.label_smoothing)
        )
        target_prob = target_prob * mask.float()
        raw_loss = self.loss_fn(
            pred_logits.transpose(1, 2), target_prob.transpose(1, 2)
        )  # [B, T]
        pred_probs = torch.softmax(pred_logits, dim=-1).detach()  # [B, T, C]

        # calculate weighted loss
        GD = (pred_probs - target_prob).abs()
        GD = torch.gather(GD, -1, target_label.unsqueeze(-1)).squeeze(-1)  # [B, T]
        GD_index = torch.floor(GD * self.num_bins).long().clamp(0, self.num_bins - 1)
        # GD = GD - 1e9 * time_mask.logical_not().float()
        weights = torch.sqrt(
            self.class_ema[target_label].detach() * self.GD_ema[GD_index].detach()
        )  # [B, T]
        loss_weighted = (raw_loss / weights) * time_mask.float()  # [B, T]
        loss_final = torch.sum(loss_weighted) / torch.sum(time_mask.float())

        if not valid:
            # update ema
            # "Elements lower than min and higher than max and NaN elements are ignored."
            target_label = (
                    target_label + (self.num_classes + 10) * time_mask.logical_not().long()
            )
            GD_index = GD_index + (self.num_bins + 10) * time_mask.logical_not().long()
            class_hist = torch.bincount(
                input=target_label.flatten(),
                weights=time_mask.flatten(),
                minlength=self.num_classes,
            )
            class_hist = class_hist[: self.num_classes]
            GD_hist = torch.bincount(
                input=GD_index.flatten(),
                weights=time_mask.flatten(),
                minlength=self.num_bins,
            )
            GD_hist = GD_hist[: self.num_bins]
            self.GD_ema = update_ema(self.GD_ema, self.alpha, self.num_bins, GD_hist)
            self.class_ema = update_ema(
                self.class_ema, self.alpha, self.num_classes, class_hist
            )

        return loss_final


if __name__ == "__main__":
    torch.manual_seed(42)
    loss_fn = MultiLabelGHMLoss(10, alpha=0.9)
    input = torch.sigmoid(torch.randn(3, 3, 10) * 10)
    target = (torch.randn(3, 3, 10) > 0).float()
    print(loss_fn(input, target))
