"""Auxiliary module for marking activations and connecting them to modules
"""
from torch import nn
import torchvision.models

ACTIVATION_MODULES = (nn.ReLU)

def mark_activations(model):
    """Marks all activation layers as activations and all other layers as non-activations 
    """
    def mark_activation(module):
        if isinstance(module, ACTIVATION_MODULES):
            module.is_activation = True
        else:
            module.is_activation = False