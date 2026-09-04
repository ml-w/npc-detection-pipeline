import ast
import logging

import click
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.controller import PMIController

from loctexthist.config.loctexthistCFG import NPCSegmentControllerCFG


@click.command()
@click.option('--inference', default=False, is_flag=True,
              help="Run in inference mode instead of training.")
@click.option('--inference-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference input directory.")
@click.option('--inference-probmap-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference probmap directory.")
@click.option('--inference-gt-dir', type=click.Path(exists=True, dir_okay=False), required=False,
              help="Override inference ground-truth directory. Must be a csv file.")
@click.option('--inference-output-dir', type=click.Path(exists=False, file_okay=False), required=False,
              help="Override the inference output directory.")
@click.option('--id-globber', type=str, default=None,
              help="Override id-globber for inference. Ignored for training.")
@click.option('--flags-file', type=click.Path(exists=True, dir_okay=False),
              default='flags.yaml',
              help="Override the flags file.")
@click.option('--id-list', default=None, type=click.Path(exists=True, dir_okay=False), required=False,
              help="If provided, overrides training/inference setting to id list.")
@click.option('--skip-posttrain-inference', default=None, is_flag=True, required=False,
              help="If specified, skip post-train inference.")
def main(inference, inference_dir, inference_probmap_dir, inference_gt_dir, inference_output_dir,
         id_globber, flags_file, id_list, skip_posttrain_inference):

    cfg = NPCSegmentControllerCFG()

    # Suppress duplicate log output caused by guild's output capture
    cfg.verbose = False

    # Standardize log format across all loggers
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handler in logger_instance.handlers:
                handler.setFormatter(formatter)

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    if inference:
        cfg.run_mode = 'inference'

        if inference_dir is not None:
            cfg.id_list = None
            cfg.data_loader_cfg.input_dir = str(inference_dir)
            cfg.data_loader_cfg.probmap_dir = None or str(inference_probmap_dir)
            cfg.data_loader_cfg.target_dir = None
        else:
            if inference_probmap_dir is not None:
                cfg.data_loader_cfg.probmap_dir = str(inference_probmap_dir)

        if inference_gt_dir is not None:
            cfg.data_loader_cfg.target_dir = str(inference_gt_dir)

        if inference_output_dir is not None:
            cfg.output_dir = inference_output_dir

    if id_list is not None:
        cfg.id_list = str(id_list)
        cfg.id_list_val = str(id_list)
        cfg.data_loader_cfg.id_list = str(id_list)

    controller = PMIController(cfg)
    controller.override_cfg(flags_file)

    # guild writes list/dict flags as strings; SegmentationSolver.decay_optimizer accesses
    # class_weights and sigmoid_params directly without ast.literal_eval (unlike lr_sche_kwargs),
    # so we must coerce them here before the solver is constructed.
    for _attr in ('class_weights', 'sigmoid_params'):
        _val = getattr(controller.solver_cfg, _attr, None)
        if isinstance(_val, str):
            setattr(controller.solver_cfg, _attr, ast.literal_eval(_val))

    if id_globber is not None:
        controller._logger.info(
            f"Overriding cfg.id_globber: {cfg.data_loader_cfg.id_globber} -> {id_globber}"
        )
        controller.data_loader_cfg.id_globber = id_globber

    controller.exec()

    # Run inference automatically after training completes
    if not inference and not skip_posttrain_inference:
        MNTSLogger.global_logger.info("Running inference.")
        controller.cfg.run_mode = 'inference'
        controller = PMIController(controller.cfg)
        controller.exec()


if __name__ == '__main__':
    main()
