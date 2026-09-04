import sys

from pytorch_med_imaging.controller import PMIController
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from pytorch_med_imaging.pmi_data_loader import (PMIImageDataLoaderCFG, PMIImageDataLoader,
                                                 PMITorchioDataLoader, PMITorchioDataLoaderCFG)
from rAIdiologist.config import rAIdiologistCFG
from rAIdiologist.config.network import *
from rAIdiologist.config.loss import *
from rAIdiologist.rai_main import *
from rAIdiologist.rai_controller import rAIController

import yaml
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import copy
from mnts.mnts_logger import MNTSLogger

# This is now the network is created
global rai_options

@click.command()
@click.option('--inference', default = False, is_flag = True , help = "For guild operation")
@click.option('--ddp'      , default = False, is_flag = True , help = "For guild operation")
@click.option('--pretrain' , default = False, is_flag = True , help = "For guild operation")
@click.option('--inference-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference directory.")
@click.option('--inference-probmap-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference probmap directory.")
@click.option('--inference-gt-dir', type=click.Path(exists=True, dir_okay=False), required=False,
              help="Override inference ground-truth directory. Must be a csv file.")
@click.option('--inference-output-dir', type=click.Path(exists=False, file_okay=False), required=False,
              help="Orveride the inference output directory")
@click.option('--id-globber', type=str, default=None,
              help="Override id-globber for inference. Ignored for training.")
@click.option('--flags-file', type=click.Path(exists=True, dir_okay=False), default='flags.yaml',
              help="Override the flags file.")
@click.option('--plotter/--no-plotter', default=True,
              help="Use --no-plotter to deactivate plotter activitiy.")
@click.option('--id-list', default=None, type=click.Path(exists=True, dir_okay=False), required=False,
              help="If provided will override training/inference setting to id list")
@click.option('--skip-posttrain-inference', default=False, is_flag=True, required=False,
              help="If specified, skip the automatic inference pass that runs after training completes.")
def main(inference, ddp, pretrain, inference_dir, inference_gt_dir, inference_probmap_dir, inference_output_dir,
         id_globber, flags_file, plotter, id_list, skip_posttrain_inference):
    controller_cls = rAIController
    if not pretrain:
        cfg = rAIdiologistCFG.MyControllerCFG()
    else:
        cfg = rAIdiologistCFG.PretrainControllerCFG()


    # Guild is giving us trouble for double printing everything, but we can't suppress it's message because it
    # won't capture the scalar values if we do that. So we have to suppress our own logger instead...
    cfg.verbose = False

    # But first, Set the log format
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    # This sets the format for guild's output
    for handlers in logging.getLogger().handlers:
        handlers.setFormatter(formatter)

    if inference:
        if ddp:
            msg = "Inference mode can't run with DDP mode."
            raise ArithmeticError(msg)
        # Put mode into inference
        cfg.run_mode = 'inference'
        cfg.plotter_init_meta['name'] += "-inference"

        # override original data directory setting if force inference instead of doing testing set evaluation
        if inference_dir is not None:
            # Note: as these are not overriden by flags file, we can change it here.
            cfg.id_list = None

            # Special branch to handle raidiologist and scdense
            if isinstance(cfg.data_loader_cfg, PMITorchioDataLoaderCFG):
                cfg.data_loader_cfg.input_data = {
                    'input': str(inference_dir),
                    'probmap': None or str(inference_probmap_dir),
                    'gt': None or str(inference_gt_dir)
                }
            else:
                # Remove idlist limitation
                cfg.data_loader_cfg.input_dir = str(inference_dir)
                cfg.data_loader_cfg.probmap_dir = None or str(inference_probmap_dir)

                # Remove gt setting
                cfg.data_loader_cfg.gt_dir = None or str(inference_gt_dir)
            cfg.data_loader_cfg.target_dir = None
        else:
            # If CFG specified directories are remained used but want to test different
            # segmentation reference, this is still available
            if inference_probmap_dir is not None:
                cfg.data_loader_cfg.probmap_dir = None or str(inference_probmap_dir)


        if inference_output_dir is not None:
            # change output dir as well
            cfg.output_dir = inference_output_dir

    # If provided as option override it
    if id_list is not None:
        cfg.id_list = str(id_list)
        cfg.id_list_val = str(id_list)
        cfg.data_loader_cfg.id_list = str(id_list)

    # If pretrain, force mode open to 0, this was done in CFG already but just incase its not loaded properly
    if pretrain:
        # Change the flags file
        with open(flags_file, 'r') as f:
            loaded_flags = yaml.safe_load(f.read())
        loaded_flags['solver_cfg']['rAI_fixed_mode'] = 0

        # TODO: Fix - This overwrites the loaded flags file unintentionally
        with open(flags_file, 'w') as f:
            yaml.dump(loaded_flags, f)

    if not plotter:
        cfg.plotting = False

    # If DDP mode is not on, simply execute one process
    if not ddp:
        controller: PMIController = controller_cls(cfg)
        controller.override_cfg(flags_file) # override CFG with flags file

        # Turn off verbosity so that we don't double print
        MNTSLogger.set_global_verbosity(False)

        if id_globber is not None:
            controller._logger.info(f"Overriding cfg.id_globber with cli command {cfg.data_loader_cfg.id_globber} "
                                    f"-> {id_globber}")
            # We need to chnage the controller's cfg instance as it's already materialized above.
            controller.data_loader_cfg.id_globber = id_globber

        # Note that default CFG shoudl points to rAI style models, which uses the BCE loss
        controller.solver_cfg.net = rai_options['networks'][controller.net_name]()
        if controller.solver_cfg.rAI_classification:
            # When in pretrain mode (i.e., mode = 0), the solver needs to change,
            # if larger than 1, we can't use binary classification solver
            controller._logger.info("Forcing loss function to be CrossEntropyLoss")
            controller.solver_cfg.loss_function = ConfidenceCELoss(weight=torch.FloatTensor([0.5, 1.2]), lambda_1=0.05, lambda_2=0.)
        controller.exec()
    else:
        # run DDP
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            msg = "DDP mode require more than one CUDA device."
            raise ArithmeticError(msg)

        mp.spawn(DDP_helper.ddp_helper, args=(world_size, copy.deepcopy(cfg), flags_file,), nprocs=world_size)

    # run inference after training
    if not inference and not skip_posttrain_inference:
        controller.cfg.run_mode = 'inference'
        controller.cfg.plotter_init_meta['name'] += "-inference"
        controller = controller_cls(controller.cfg)
        if plotter:
            controller._plotter.add_tag(f"{model}")
        controller.exec()

if __name__ == '__main__':
    main()