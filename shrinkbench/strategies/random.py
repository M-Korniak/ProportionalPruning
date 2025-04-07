"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline for pruning evalution
"""

import numpy as np
from ..pruning import VisionPruning, LayerPruning
from .utils import map_importances


def random_mask(tensor, fraction):
    idx = np.random.uniform(0, 1, size=tensor.shape) > fraction
    mask = np.ones_like(tensor)
    mask[idx] = 0.0
    return mask


class RandomPruning(VisionPruning):

    def model_masks(self):
        params = self.params()
        masks = map_importances(lambda x: random_mask(x, self.fraction), params)
        return masks


class LayerRandomStructured(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        masks = {}
        if params['weight'].ndim == 2:
            # This is a linear layer
            num_to_select = int(self.fraction * params['weight'].shape[0])
            selected_indices = np.random.choice(params['weight'].shape[0], size=num_to_select, replace=False)

            # Create mask
            mask = np.zeros_like(params['weight'], dtype=bool)
            mask[selected_indices] = True
            masks = {'weight': mask}

            # Same indices for biases
            if params['bias'] is not None:
                masks['bias'] = np.zeros_like(params['bias'], dtype=bool)
                masks['bias'][selected_indices] = True

        elif params['weight'].ndim == 4:
            # This is a Conv2d layer
            num_to_select = int(self.fraction * params['weight'].shape[0])  # num output channels
            selected_indices = np.random.choice(params['weight'].shape[0], size=num_to_select, replace=False)

            # Create mask to prune out channels
            mask = np.zeros_like(params['weight'], dtype=bool)
            mask[selected_indices] = True
            masks['weight'] = mask

            # Same indices for biases
            if params['bias'] is not None:
                masks['bias'] = np.zeros_like(params['bias'], dtype=bool)
                masks['bias'][selected_indices] = True

        else:
            raise ValueError(f"Module {module} has weight of shape {params['weight'].shape}")

        return masks
