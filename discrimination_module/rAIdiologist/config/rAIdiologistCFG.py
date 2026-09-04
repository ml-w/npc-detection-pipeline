"""
rAIdiologistCFG.py — Configuration classes for the rAIdiologist pipeline.

This module defines CFG objects consumed by PMIController.override_cfg() at runtime.
Each CFG class wires together: data loaders, the network, solver, loss, and controller
settings for a specific training mode.

Training modes (rAI_fixed_mode):
    0  — pretrain: CNN trains, RNN frozen (standalone CNN classification)
    1  — CNN fine-tune: CNN trains, RNN frozen (rnn.out_fc stays learnable)
    2  — RNN training: RNN trains, CNN frozen (eval mode)
    3  — joint: both CNN and RNN train simultaneously
   -1  — inference: all gradients off, full eval
   None — auto-schedule: 0–33% → mode 2, 33–66% → mode 1, 66–100% → mode 3

CFG hierarchy:
    PMIControllerCFG
    └── MyControllerCFG            (standard rAI training, fold B00 by default)
        ├── PretrainControllerCFG  (overrides for CNN pretraining)
        └── rAIControllerFocusedCFG
            └── FocusedPretrainControllerCFG

Data loaders are instantiated at module level (data_loader, data_loader_test,
data_loader_focused, data_loader_focused_test) and referenced by the CFG classes.
Guild flags (flags.yaml) are applied by override_cfg() after this module is imported,
so env-name flags must be used for any attribute read at class-body evaluation time.
"""

from pytorch_med_imaging.controller import PMIControllerCFG
from pytorch_med_imaging.pmi_data_loader import (PMIImageFeaturePairLoader, PMIImageFeaturePairLoaderCFG,
                                                 PMITorchioDataLoader, PMITorchioDataLoaderCFG)
from .network.rAIdiologist import rAIdiologist
from .loss.rAIdiologist_loss import ConfidenceBCELoss
from ..solvers.rAIdiologistSolver import rAIdiologistSolverCFG, rAIdiologistSolver, rAIdiologistInferencer
from datetime import datetime
from typing import *
import torch
import os

# For training
data_loader = PMIImageFeaturePairLoaderCFG(
    input_dir     = None,   # Nyul-normalised T2WFS images; set via flags.yaml or guild flag
    probmap_dir   = None,   # HuangThresholding tissue masks; set via flags.yaml or guild flag
    target_dir    = None,   # label CSV with is_malignant column; set via flags.yaml or guild flag
    augmentation  = './rAIdiologist_transform_train.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    sampler       = 'weighted', # Unset sampler to load the whole image
    sampler_kwargs    = dict(
        patch_size = [320, 320, 25] # Original paper size is 320 x 320 x 25, save memory here for debugging
    ),
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 240,
        samples_per_volume     = 2,
        num_workers            = min(12, os.cpu_count() * 3 // 4),
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

# For testing
data_loader_test = PMIImageFeaturePairLoaderCFG(
    input_dir     = data_loader.input_dir,
    probmap_dir   = data_loader.probmap_dir,
    target_dir    = data_loader.target_dir,
    target_column = data_loader.target_column,
    id_globber    = data_loader.id_globber,
    augmentation  = './rAIdiologist_transform_inf.yaml',
    tio_queue_kwargs = dict(            # `dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = min(12, os.cpu_count() * 3 // 4),
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)


class  MySolverCFG(rAIdiologistSolverCFG):
    r"""This is created to cater for the configuration of rAIdiologist network"""
    net           = rAIdiologist(out_ch = 1, cnn_dropout= 0.2, rnn_dropout= 0.2)
    rAI_run_mode  = 1
    optimizer     = 'Adam'
    init_lr       = 1E-4
    batch_size    = 8
    num_of_epochs = 200

    unpack_key_forward   = ['input'  , 'gt']
    unpack_key_inference = ['input']

    early_stop        = 'loss_reference'
    early_stop_kwargs = {'warmup'       : 5, 'patience': 10}
    accumulate_grad   = 0

    # This make the inference saves the transformer playbacks when model is rAI
    rAI_inf_save_playbacks = True

    loss_function = ConfidenceBCELoss(pos_weight = torch.as_tensor([float(os.environ.get("RAI_LOSS_POS_WEIGHT", 1.2))]),
                                      conf_weight=float(os.environ.get("RAI_CONF_WEIGHT", 0.2)),
                                      over_conf_weight=float(os.environ.get("RAI_OVER_CONF_WEIGHT", 0.5)),
                                      gamma=float(os.environ.get("RAI_GAMMA", 0.5)))


class MyControllerCFG(PMIControllerCFG):
    run_mode    = 'training'
    fold_code   = 'B01'
    id_list     = None
    id_list_val = None
    output_dir  = './Results/{fold_code}'
    cp_load_dir = './Backup/rAIdiologist_{fold_code}.pt'
    cp_save_dir = './Backup/rAIdiologist_{fold_code}.pt'
    log_dir     = f"./Backup/Log/rAIdiologist_{datetime.strftime(datetime.now(), '%Y-%m-%d')}.log"
    rAI_pretrained_CNN = './Backup/rAIdiologist_{fold_code}_pretrain.pt'

    _data_loader_cfg     = data_loader
    _data_loader_inf_cfg = data_loader_test # inference need different dataloader
    data_loader_val_cfg  = data_loader_test # validation set comes from the same folder as testing set
    data_loader_cls      = PMIImageFeaturePairLoader

    solver_cfg     = MySolverCFG()
    solver_cls     = rAIdiologistSolver
    inferencer_cls = rAIdiologistInferencer

    debug_mode = True
    debug_validation = False
    compile_net = False

    # For plotting
    plotting          = True
    plotter_type      = 'wandb'
    plotter_init_meta = { # Use wandb
        'entity' : "lun-m-wong-cuhk",
        'project': "NPC-Screening",
        'name': "rAIdiologist",
        'notes'  : "rAIdiologists training project.",
    }


class PretrainControllerCFG(MyControllerCFG):
    # solver_cls     = BinaryClassificationSolver
    # inferencer_cls = BinaryClassificationInferencer
    cp_load_dir    = MyControllerCFG.cp_load_dir.replace('.pt', '_pretrain.pt')
    cp_save_dir    = MyControllerCFG.cp_save_dir.replace('.pt', '_pretrain.pt')
    output_dir     = MyControllerCFG.output_dir + "_pretrain"
    # Override some settings
    solver_cfg     = MySolverCFG()
    solver_cfg.rAI_pretrain_mode = True
    solver_cfg.rAI_fixed_mode = 0

    plotter_init_meta = {
        'entity' : "lun-m-wong-cuhk",
        'project': "NPC-Screening",
        'name': "rAIdiologist-Pretrain",
        'notes'  : "rAIdiologists pre-training project.",
    }