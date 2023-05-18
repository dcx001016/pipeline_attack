import random
import torch

def attack_hook(backward_attack_rate):
    def hook(module, grad_input, grad_output):
        p = random.random()
        grad_input0 = grad_input[0]
        abs_x = grad_input0.abs()
        mean = abs_x.mean().item()
        max = abs_x.max().item()
        min = abs_x.min().item()
        median = abs_x.median().item()
        print("backward grad:", mean, max, min, median)
        if p > 1 - backward_attack_rate:
            perturbation = torch.randn_like(grad_input0)
            # grad_input0 *= -1 * p
            grad_input0 += perturbation
            grad_input = tuple([grad_input0])
        return grad_input
    return hook