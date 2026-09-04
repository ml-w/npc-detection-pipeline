import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.nn.utils.rnn import *
from torch.utils.hooks import RemovableHandle

from pytorch_med_imaging.loss import BinaryFocalLoss
import os
import pprint
from typing import Any, Callable, Dict, Optional, Tuple, Union

__all__ = ['ConfidenceBCELoss', 'ConfidenceCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""Computes the custom classification loss combining multiple loss components.

    This loss function combines a base classification loss with confidence-aware
    regularization terms. It handles both standard predictions (single column input)
    and confidence-based predictions (multi-column input). For multi-column inputs,
    the loss combines:

    .. math::
        \mathcal{L} = \mathcal{L}_{base} + \alpha \mathcal{L}_{conf} + \beta \mathcal{L}_{overconf} + \gamma \mathcal{L}_{reg}

    where:

    .. math::
        \mathcal{L}_{base} &= BCE(predictions, target) \\
        \mathcal{L}_{conf} &= BCE(sig\_conf, correctness) \\
        \mathcal{L}_{overconf} &= E[sig\_conf \cdot (1 - correctness)] \\
        \mathcal{L}_{reg} &= BCE(sig\_conf, 0.5) \\
        correctness &= sig\_rnn\_pred \cdot target + (1 - sig\_rnn\_pred) \cdot (1 - target)


    .. note::
        The confidence is now embedded as weights, which guide the loss to be small for initial
        slices, and then increases the weight for later slices. The delay of this increase in
        weight can be controlled using the parameter 'delay'

    """
    def __init__(self, *args, conf_weight=0.3, over_conf_weight=0.5, gamma=0.1, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        # self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
        #                                       pos_weight=torch.FloatTensor([conf_pos_weight]))

        # self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        # Number of slices before the decision becomes important (in unit number of slices)
        # self.register_buffer('_delay', torch.DoubleTensor([delay]))
        self.register_buffer('alpha', torch.DoubleTensor([conf_weight]))
        self.register_buffer('beta', torch.DoubleTensor([over_conf_weight]))
        self.register_buffer('gamma', torch.DoubleTensor([gamma]))
        # self.register_buffer('_sigma', torch.DoubleTensor([2.]))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the custom classification loss.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, 2) where input[:, 0] are predictions and input[:, 1] are confidence scores.
            target (torch.Tensor): Ground truth tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Computed loss value.
        """
        batch_size = input.size(0)
        # If input has only one output, use the base loss
        if input.shape[1] == 1:
            return self.base_loss(input, target).mean()
        else:
            predictions, confidence, rnn_predictions = input[:, 0], input[:, 1], input[:,2]

            sig_rnn_pred = torch.sigmoid(rnn_predictions)
            sig_conf = confidence # This is expected to have already been sigmoided

            # Clamp to prevent numerical issues
            epsilon = 1e-7
            sig_rnn_pred = torch.clamp(sig_rnn_pred, epsilon, 1 - epsilon)
            sig_conf = torch.clamp(sig_conf, epsilon, 1 - epsilon)

            # Base loss (i.g., Binary Cross-Entropy with Logits)
            base_loss = self.base_loss(predictions, target)

            # For validation mode, we don't need other terms for determining stopping point
            if not input.requires_grad:
                return base_loss.mean()

            # Soft correctness: higher when prediction aligns with target
            # For binary classification: target in {0,1}
            correctness = sig_rnn_pred * target + (1 - sig_rnn_pred) * (1 - target)  # Probability of being correct

            # Confidence regularization: encourage sig_conf to match correctness
            confidence_loss = F.binary_cross_entropy(sig_conf, correctness)

            # Overconfidence penalty: discourage high confidence when correctness is low
            overconfidence_penalty = torch.mean(sig_conf * (1 - correctness))  # High when confident but incorrect

            # Encourage some level of confidence (e.g., prevent all confidences from being too low)
            regularization = F.binary_cross_entropy(sig_conf, torch.full_like(sig_conf, 0.5))

            # Combine all loss components
            loss = base_loss + self.alpha * confidence_loss + \
                   self.beta * overconfidence_penalty + self.gamma * regularization
            return loss.mean()


    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x


class ConfidenceCELoss(nn.Module):
    def __init__(self, *args, lambda_1=0.01, lambda_2=0.3,  **kwargs):
        r"""This is essentially the CrossEntropy version of the above binary prediction loss.

        Features:
            - Regularizer term

        .. math::
            \operatorname{Loss}(X, Y) = \frac{1}{N}\sum_s^N \text{sigmoid}(s;d)

        Args:
            lambda_1 (float, Optional): Regularizer for CNN:RNN weight, encourage it to be far from 0.5
            lambda_2 (float, Optional): Regularizer for X, encourage it to be non zero.

            *args (Any): Arguments for base_loss.
            **kwargs (Any): Keyword arguments for base_loss.

        """
        super(ConfidenceCELoss, self).__init__()
        self.register_buffer('lambda_1', torch.DoubleTensor([lambda_1]))
        self.register_buffer('lambda_2', torch.DoubleTensor([lambda_2]))
        self.base_loss = nn.CrossEntropyLoss(*args, **kwargs)


    def forward(self,
                input: Tuple[torch.FloatTensor, torch.FloatTensor],
                target: torch.LongTensor) -> torch.FloatTensor:
        r"""

        .. note::
            * There are no sigmoid/softwmax needed for the prediction, but the confidence value should have be
              sigmoided within range of 0 to 1.

        Args:
            input (tuple):
                Input should be a tuple of two tensors, first on is the prediction, second is the confidence.
                The inputs should have a size :math:`(B \times C)` and :math:`(B \times 1)`, respectively.
            target (torch.LongTensor):
                Target class

        Returns:
            torch.FloatTensor: Loss
        """
        # Unpack
        try:
            # if input contains multiple elements
            model_pred, model_conf = input

            # Get baseloss
            base_loss = self.base_loss(model_pred, target)

            # calculate regularization losses
            rho = 0.2
            lam1 = self.lambda_1 * (torch.exp(-(model_conf-0.5).div(rho).pow(2))).mean() # penalize conf near 0.5
            # lam2 = self.lambda_2 * (- torch.sigmoid((model_pred.abs())) + 1).mean() # penalize pred near zero
            return base_loss + lam1
        except ValueError:
            # otherwise its just an ordinary CE loss
            return self.base_loss(input, target)

    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x
