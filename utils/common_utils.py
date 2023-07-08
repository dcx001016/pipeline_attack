import torch
import wandb
from communication.comm_utils import get_pipeline_parallel_rank

def print_tensor(x: torch.Tensor, description):
    abs_x = x.abs()
    mean = abs_x.mean().item()
    max = abs_x.max().item()
    min = abs_x.min().item()
    median = abs_x.median().item()
    print(f"{description} tensor size: {tuple(x.size())} mean: {mean:.6f} max: {max:.6f} min: {min:.6f} median: {median:.6f}")

def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def init_wandb_config(args):
    config = wandb.config
    for attr, value in vars(args).items():
        setattr(config, attr, value)

def wandb_activation_gradient(activation: torch.Tensor, gradient: torch.Tensor):
    return
    wandb.log({
            'activation_mean': activation.mean().item(),
            'activation_std': activation.std().item(),
            'activation_max': activation.max().item(),
            'activation_min': activation.min().item(),
            'activation_median': activation.median().item(),
            'gradient_mean': gradient.mean().item(),
            'gradient_std': gradient.std().item(),
            'gradient_max': gradient.max().item(),
            'gradient_min': gradient.min().item(),
            'gradient_median': gradient.median().item(),
        })

def print_tensor_gradient(grad):
    print_tensor(grad, f"rank: {get_pipeline_parallel_rank()} tensor_gradient: ")

def print_gradient(module, grad_input, grad_output):
    grad_input0 = grad_input[0]
    grad_output0 = grad_output[0]
    print_tensor(grad_input0, f"rank: {get_pipeline_parallel_rank()} grad_input: ")
    print_tensor(grad_output0, f"rank: {get_pipeline_parallel_rank()} grad_output: ")

def print_activation(module, input):
    input0 = input[0]
    print_tensor(input0, f"rank: {get_pipeline_parallel_rank()} activation_input: ")