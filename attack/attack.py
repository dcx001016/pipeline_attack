import random
import torch
from typing import Tuple

# apply register_hook to the last layer of the model on each virtual GPU except the last one.
def attack_backward_hook(backward_attack_rate):
    def hook(grad_input: torch.Tensor):
        p = random.random()
        if p > 1 - backward_attack_rate:
            # grad_input *= -1 * p
            # perturbation = torch.randn_like(grad_input)
            perturbation = torch.normal(mean=float(grad_input.mean()), std=float(grad_input.std()), size=tuple(grad_input.shape)).to(grad_input.device)
            grad_input += perturbation
        return grad_input
    return hook

# apply register_forward_pre_hook to the first layer of the model on each virtual GPU except the first one.
def attack_forward_hook(forward_attack_rate):
    def hook(module, input: Tuple[torch.Tensor]):
        if module.training:
            p = random.random()
            if p > 1 - forward_attack_rate:
                input0 = input[0]
                # input0 = input0 * -1 * p
                # perturbation = torch.randn_like(input0)
                perturbation = torch.normal(mean=float(input0.mean()), std=float(input0.std()), size=tuple(input0.shape)).to(input0.device)
                input0.data.add_(perturbation)
                input = tuple([input0])
                return input
    return hook