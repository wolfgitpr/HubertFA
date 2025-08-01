from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from .chained_optimizer import ChainedOptimizer, OptimizerSpec


def get_bf16_support_map():
    bf16_support_map = {}

    if not torch.cuda.is_available():
        return bf16_support_map

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return bf16_support_map

    for i in range(device_count):
        device = torch.device(f'cuda:{i}')
        major, minor = torch.cuda.get_device_capability(device)
        bf16_support_map[device] = (major >= 8)

    return bf16_support_map


def zeropower_via_newtonschulz5(G: Tensor, steps: int, use_bf16: bool) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 3  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    if use_bf16:
        X = G.bfloat16()
    else:
        X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.bf16_support_map = get_bf16_support_map()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["buffers"].append(buf)
            for key in shape_groups:
                group_data = shape_groups[key]
                g = torch.stack(group_data["grads"])
                buf = torch.stack(group_data["buffers"])
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if g.ndim >= 4:  # for the case of conv filters
                    g = g.view(g.size(0), g.size(1), -1)
                use_bf16 = self.bf16_support_map.get(g.device, False)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], use_bf16=use_bf16)
                for i, p in enumerate(group_data["params"]):
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.data.add_(g[i].view_as(p), alpha=-group["lr"] * max(g[i].size()) ** 0.5)
                    self.state[p]["momentum_buffer"] = buf[i].clone()


def get_params_for_muon(model) -> List[Parameter]:
    """
    Filter parameters of a module into two groups: those that can be optimized by Muon,
    and those that should be optimized by a standard optimizer.
    Args:
        module: The module to filter parameters for.
    Returns:
        A list of parameters that should be optimized with muon.
    """
    muon_params = []
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            if not isinstance(module, nn.Embedding) and param.ndim >= 2:
                muon_params.append(param)
    return muon_params


class Muon_AdamW(ChainedOptimizer):
    def __init__(self, model, lr=0.0005, weight_decay=0.0, muon_args={}, adamw_args={}, verbose=False):
        muon_params_id_set = set(id(p) for p in get_params_for_muon(model))
        spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
        spec_adamw = OptimizerSpec(torch.optim.AdamW, adamw_args, None)
        specs = [spec_muon, spec_adamw]
        callback = None
        if verbose:
            callback = lambda p, spec_idx: print(
                f"Adding param {p.shape} to optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
            )
        super().__init__(model.parameters(), specs, lr=lr, weight_decay=weight_decay,
                         optimizer_selection_callback=callback)
