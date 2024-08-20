import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

LOGGER = logging.getLogger("main")


def customized_worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # To disable hash randomization, make the experiment reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed) # Set random seed for CPU
    torch.cuda.manual_seed(seed) # Set random seed for current GPU
    torch.cuda.manual_seed_all(seed) # Set random seed for all GPUs

def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    if seed >= 0:
        LOGGER.info(f"We will use a fixed seed {seed}")
    else:
        seed = np.random.randint(2**32)
        LOGGER.info(f"We will use a random seed {seed}")
    set_seed_for_lib(seed)
    if not deterministic:
        LOGGER.info("We will use `torch.backends.cudnn.benchmark`")
    else:
        LOGGER.info("We will not use `torch.backends.cudnn.benchmark`")
    cudnn.enabled = True
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def to_device(data, device="cuda"):
    if isinstance(data, (tuple, list)):
        return [to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {name: to_device(item, device) for name, item in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    else:
        raise TypeError(f"Unsupported type {type(data)}. Only support Tensor or tuple/list/dict containing Tensors.")


def frozen_bn_stats(model, freeze_affine=False):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if freeze_affine:
                m.requires_grad_(False)
