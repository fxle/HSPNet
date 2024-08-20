# -*- coding: utf-8 -*-

import os

import torch


def save_weight(save_path, model):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(model_state, save_path)


def load_weight(load_path, model, *, strict=True, skip_unmatched_shape=False):
    assert os.path.exists(load_path), load_path
    model_params = model.state_dict()
    for k, v in torch.load(load_path, map_location="cpu").items():
        if k.endswith("module."):
            k = k[7:]
        if skip_unmatched_shape and v.shape != model_params[k].shape:
            continue
        model_params[k] = v
    model.load_state_dict(model_params, strict=strict)
