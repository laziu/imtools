import torch
import torch.nn as nn


def compare_weights(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """ Compare the torch module weights. """
    src = source.keys()
    trg = target.keys()
    return {
        "source_only": [x for x in src if x not in trg],
        "target_only": [x for x in trg if x not in src],
    }


def freeze(model: nn.Module) -> nn.Module:
    """ Freeze the weights of the model. """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model: nn.Module) -> nn.Module:
    """ Unfreeze the weights of the model. """
    for param in model.parameters():
        param.requires_grad = True
    return model
