import torch

def get_device(idle = False, config=None):
    if idle:
        return "cpu" if config is None else config["idle_device"]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)

