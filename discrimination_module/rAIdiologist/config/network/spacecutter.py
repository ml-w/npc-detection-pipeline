r"""Ordinal regression utilities.

Originally from https://github.com/EthanRosenthal/spacecutter
"""
from copy import deepcopy

import torch
from torch import nn

__all__ = ['LogisticCumulativeLink', 'OrdinalLogisticModel']


class LogisticCumulativeLink(nn.Module):
    """Converts a single scalar prediction to proportional odds over ordered classes.

    Args:
        num_classes (int):
            Number of ordered classes (must be > 2).
        init_cutpoints (str):
            How to initialise the cut-points. ``'ordered'`` (default) spaces them
            evenly via a sigmoid; ``'random'`` draws them uniformly at random.
    """

    def __init__(self, num_classes: int, init_cutpoints: str = 'ordered') -> None:
        assert num_classes > 2, 'Only use this model if you have 3 or more classes'
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            cutpoints = (torch.sigmoid(cutpoints * self.num_classes) - 0.5) * self.num_classes
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Equation (11) from *On the consistency of ordinal regression methods*, Pedregosa et al."""
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
            sigmoids[:, [0]],
            link_mat,
            (1 - sigmoids[:, [-1]])
        ), dim=1)
        return link_mat


class OrdinalLogisticModel(nn.Module):
    """Wraps any single-output predictor with :class:`LogisticCumulativeLink`.

    Args:
        predictor (nn.Module):
            A module that returns a ``FloatTensor`` of shape ``(batch_size, 1)``.
        num_classes (int):
            Number of ordered classes.
        init_cutpoints (str):
            Passed to :class:`LogisticCumulativeLink`. Default ``'ordered'``.
    """

    def __init__(self, predictor: nn.Module, num_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.predictor = deepcopy(predictor)
        self.link = LogisticCumulativeLink(self.num_classes, init_cutpoints=init_cutpoints)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.link(self.predictor(*args, **kwargs))
