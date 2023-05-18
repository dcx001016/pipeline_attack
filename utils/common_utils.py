import wandb


def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def init_wandn_config(args):
    config = wandb.config
    for attr, value in vars(args).items():
        setattr(config, attr, value)
