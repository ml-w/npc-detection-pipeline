import os
import copy
import torch.distributed as dist
from .config.network import *
from .config.network.old.old_swran import SlicewiseAttentionRAN_old
from .rai_controller import rAIController
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG

global rai_options
# Values are callables — instantiated on demand, not at import time.
rai_options = {
    'networks': {
        # Developed networks (Note lambda is need to prevent initing a large amount of networks)
        'rai_v5.1'          : create_rAIdiologist_v5_1,
        'rai_v5.1-focused'  : create_rAIdiologist_v5_1_focused,
        # Need to do lambda to avoid initializing network here
        'mean_swran'        : lambda: SlicewiseAttentionRAN_old(1, 1, reduce_by_mean=True),
        'old_swran'         : lambda: SlicewiseAttentionRAN_old(1, 1),
        'new_swran'         : lambda: SlicewiseAttentionRAN(1, 1, dropout=0, reduce_strats='max'),

        # Baseline networks
        'resnet3d10'        : lambda : get_ResNet3d(10),
        'resnet3d18'        : lambda : get_ResNet3d(18),
        'resnet3d34'        : lambda : get_ResNet3d(34),
        'resnet3d50'        : lambda : get_ResNet3d(50),
        'resnet3d101'       : lambda : get_ResNet3d(101),
        'resnet3d152'       : lambda : get_ResNet3d(152),
        'resnet3d200'       : lambda : get_ResNet3d(200),
        'vgg16'             : lambda : get_vgg('16'),
        'vgg11'             : lambda : get_vgg('11'),
        'densenet3d121'     : lambda : get_densenet3d('121'),
        'densenet3d169'     : lambda : get_densenet3d('169'),
        'densenet3d201'     : lambda : get_densenet3d('201'),
        'densenet3d264'     : lambda : get_densenet3d('264'),
        'vit'               : lambda : ViT(1, 1, num_slices=25),
    }
}

class DDP_helper:
    r"""A helper class for distributed data parallel (DDP) training using PyTorch. This class provides a function
    `ddp_helper` that initializes the DDP process group and creates a `rAIController` instance for each process.
    The `rAIController` instance is used to run the training loop on each process, with instance templates passed
    as class attributes. The batch size is adjusted for each process to ensure that the total batch size remains
    the same.
    """
    @classmethod
    def ddp_helper(cls, rank: int, world_size: int, cfg: dict, flags: PMIControllerCFG):
        r"""
        Args:
            rank (int):
                The rank of the current process.
            world_size (int):
                The total number of processes.
            cfg (dict):
                A dictionary containing the configuration options for the `rAIController` instance.
            flags (PMIControllerCFG):
                An object containing the command-line arguments for the training script.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "23455"
        dist.init_process_group("nccl", world_size=world_size, rank=rank)

        controller = rAIController(cfg)
        controller.override_cfg(flags)
        # Change the batch-size because each controller only has one GPU
        controller.solver_cfg.batch_size = controller.solver_cfg.batch_size // world_size
        # Override the network setting
        controller.solver_cfg.net = rai_options['networks'][controller.net_name]()
        controller.exec()

        dist.barrier()
        dist.destroy_process_group()