"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

import numpy as np

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)


class GlobalMagWeight(VisionPruning):

    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagWeight(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        print(module)
        params = self.module_params(module)
        importances = {param: np.abs(value) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks

class LayerL1Stuctured(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        masks = {}
        if params['weight'].ndim == 2:
            # this is linear
            # Calculate first norm per row
            norms = np.linalg.norm(params['weight'], ord=1, axis=1)
            # Sort by norm descending
            sorted_indices = np.argsort(-norms)
            # select self.fraction * len(norms) indices
            num_to_select = int(self.fraction * len(norms))
            selected_indices = sorted_indices[:num_to_select]
            # Create mask
            mask = np.zeros_like(params['weight'], dtype=bool)
            mask[selected_indices] = True
            masks = {'weight': mask}
            # the same indices for biases
            if params['bias'] is not None:
                masks['bias'] = np.zeros_like(params['bias'], dtype=bool)
                masks['bias'][selected_indices] = True

        elif params['weight'].ndim == 4:
            # This is a Conv2d layer
            # Calculate L1 norm per channel (axis 1 in weight is channel dimension)
            norms = np.sum(np.abs(params['weight']), axis=(1, 2, 3)) # L1 norm across height, width, and channels
            sorted_indices = np.argsort(-norms)
            num_to_select = int(self.fraction * len(norms))  # Fraction of channels to keep
            selected_indices = sorted_indices[:num_to_select]
            
            # Create mask to prune out channels
            mask = np.zeros_like(params['weight'], dtype=bool)
            mask[selected_indices] = True
            masks['weight'] = mask
            
            # The same indices for biases (if applicable)
            if params['bias'] is not None:
                masks['bias'] = np.zeros_like(params['bias'], dtype=bool)
                masks['bias'][selected_indices] = True
        else:
            raise ValueError(f"Module {module} has weight of shape {params['weight'].shape}")
        
        return masks


class GlobalMagGrad(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(params[mod][p]*grads[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagGrad(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        importances = {param: np.abs(value*grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagAct(ActivationMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        activations = self.activations()
        # [0] is input activation
        importances = {mod:
                       {p: np.abs(activation_importance(params[mod][p], activations[mod][0]))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagAct(ActivationMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        input_act, _ = self.module_activations(module)
        importances = {param: np.abs(activation_importance(value, input_act))
                       for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks
