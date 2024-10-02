import torch


def dcn(x):
    """Tensor to numpy: Detach-Cpu-Numpy."""
    if isinstance(x, dict):
        return {k: dcn(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return tuple(dcn(v) for v in x)
    elif not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()
