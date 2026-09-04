import pprint
from pathlib import Path

import click
from pytorch_med_imaging.pmi_data_loader import PMITorchioDataLoader, PMITorchioDataLoaderCFG
from rAIdiologist.solvers import rAIdiologistSolverCFG, rAIdiologistInferencer
from rAIdiologist.config.network.rAIdiologist import *
from rAIdiologist.rai_controller import PMIControllerCFG, rAIController

from mnts.mnts_logger import MNTSLogger


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

@click.command()
@click.option('--image-data-dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help='Directory for image data. All files with .nii.gz suffix are globbed.',
              required=True)
@click.option('--probmap-dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help='Directory for segmentation data with .nii.gz suffix, the ID should pair with data found in provided'
                   ' image-data-dir.',
              required=True)
@click.option('--checkpoint-dir',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              default=Path("/home/lwong/Source/Repos/NPC_Segmentation/Backup/rAIdiologist_B01-c6b89d35.pt"),
              help='Path to the checkpoint file.',
              required=True)
@click.option('--output-dir',
              type=click.Path(file_okay=False, writable=True, dir_okay=True, path_type=Path),
              default=Path("/mnt/ftp/Shared/temp"),
              help='Directory for output files.',
              required=True)
@click.option('--id-globber',
              type=str,
              default=r'\w{0,5}\d+',
              help='ID globbing pattern.')
@click.option('--id-list',
              required=False,
              multiple=True,
              default=None,
              help='List of IDs.')
@click.option('--id-list-file',
              required=False,
              default=None,
              type=click.Path(file_okay=False, writable=True, dir_okay=True, path_type=Path),
              help='ID List file, if this is specified, id-list will be ignored.')
@click.option('--inference-transform',
              type=click.Path(file_okay=True, exists=True, path_type=Path),
              default="../rAIdiologist/config/rAIdiologist_transform_inf.yaml",
              help='Path to inference transformation configuration file.')
@click.option('--ground-truth',
              type=click.Path(file_okay=True, exists=True),
              default=None,
              help="Not implemented yet.",
              required=False)
@click.option('--debug',
              is_flag=True,
              help="For debugging.")
def main(**kwargs):
    r"""Easy-to-use cli for inferencing the network.

    Args:
        **kwargs:

    Returns:

    """
    args = Args(**kwargs)

    with (torch.no_grad(),
        MNTSLogger(".", "PlaybackTest", verbose=True, log_level='debug',
                     keep_file=False) as logger):
        logger.debug(f'f{args = }')

        # -- update the cfg with input parameters
        # Setup dataloader CFG
        data_loader_cfg = PMITorchioDataLoaderCFG(
            input_data = { # TODO: implement ground-truth for auto analysis
                'input' :   args.image_data_dir,
                'probmap' : args.probmap_dir

            },
            input_dtypes = {
                'probmap': 'uint8'
            },
            master_data_key='input',
            augmentation=args.inference_transform,
            ignore_missing_ids=True,
            sampler='weighted',
            sampler_kwargs = dict(
                patch_size = [320,320,25]
            ),
            tio_queue_kwargs=dict(  # `dict passed to ``tio.Queue``
                max_length         = 15,
                samples_per_volume = 1,
                num_workers        = min(12, os.cpu_count() * 3 // 4),
                shuffle_subjects   = True,
                shuffle_patches    = True,
                start_background   = True,
                verbose            = True,

            )
        )
        # Setup Inferencer CFG
        inferencer_cfg = rAIdiologistSolverCFG(
            net                          = create_rAIdiologist_v5_1(),
            batch_size                   = 8,
            unpack_key_inference         = ['input'],
            rAI_inf_save_playbacks       = True,
            rAI_fixed_mode               = -1

        )

        # Setup Controller CFG
        # BUG: don't know why but this declartion tries to set data_loader_cfg
        controller_cfg = PMIControllerCFG(
            run_mode        = 'inference',
            fold_code       = None, # this will disable replacements``
            id_list         = args.id_list or args.id_list_file,
            cp_load_dir     = args.checkpoint_dir,
            output_dir      = args.output_dir,
            log_dir         = "./inference.log",
        )



        # -- Create the CVS
        controller = rAIController(controller_cfg)
        # note that properties needs to be set after instant creation
        controller_cfg.solver_cfg=inferencer_cfg, # Inferencer and solver use the same CFG keyword
        controller.inferencer_cls = rAIdiologistInferencer
        controller.data_loader_cls = PMITorchioDataLoader
        controller.solver_cfg = inferencer_cfg
        controller.data_loader_cfg = data_loader_cfg
        controller.exec()


if __name__ == '__main__':
    main()
    input('Press any key to continue...')