import os
import tempfile
from pathlib import Path

import SimpleITK as sitk
import click
import copy
import shutil
import re
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs, get_fnames_by_IDs
from ..config.loctexthistCFG import *
from ..utils.preprocess import NPCSegmentPreprocesser
from ..utils.img_proc import *
from .progress_manager import ProgressStateManager
import logging
from typing import Union, Any, List, Optional, Dict
import time

PathLike = Union[str, Path]

sequence_choice = ['T2WFS', 'T1W', 'CET1W', 'CET1WFS']

def create_temp_dir_for_inference(id_file: PathLike, id_globber: str,
                                  input_dir: PathLike) -> tempfile.TemporaryDirectory:
    """Creates a temporary directory with symbolic links to input files for inference.

    This function reads a list of IDs from a file, matches them with available files in the
    input directory, and creates a temporary directory containing symbolic links to the
    corresponding files.

    Args:
        id_file (PathLike):
            Path to a file containing IDs. Supports .txt (one ID per line)
            or .ini (comma-separated IDs under [FileList]->testing section) formats.
        id_globber  (str):
            Pattern string to extract IDs from filenames.
        input_dir  (PathLike):
            Directory containing the input .nii.gz files.
        output_dir  (PathLike):
            Directory where results will be saved (used for validation).

    Returns:
        tempfile.TemporaryDirectory: A temporary directory object containing symbolic links
            to the input files. The directory is automatically cleaned up when the object
            is destroyed.

    Raises:
        ValueError: If the id_file format is unknown or if any specified ID is not found
            in the input directory.

    .. note::
        Note that this does not clean the directory for you, you have to clean it yourself.

    Example:
         >>> temp_dir = create_temp_dir_for_inference(
         ...     id_file='subjects.txt',
         ...     id_globber=r'sub-(\d+)',
         ...     input_dir='raw_data',
         ...     output_dir='results'
         ... )
         >>> # Use the temporary directory
         >>> temp_dir.cleanup()
    """
    logger = logging.getLogger('global.inference')

    # Make sure the directories are Path
    id_file = Path(id_file)
    input_dir = Path(input_dir)

    # Load the specified IDs
    if id_file.suffix == '.txt':
        ids = [line.strip() for line in id_file.open('r').readlines()]
    elif id_file.suffix == '.ini':
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read(str(id_file))
        ids = parser['FileList']['testing'].split(',')
    elif str(id_file).find(',') >= 0:
        ids = str(id_file).split(',')
    else:
        raise ValueError(f'Unknown id_file format: {id_file.suffix}')

    # Read available IDs from input_dir
    avail_ids = get_unique_IDs(input_dir.rglob('*.nii.gz'), id_globber, return_dict=True)
    if len(avail_ids) == 0:
        logger.debug(f"{','.join([r.name for r in input_dir.rglob('*')])}")
        raise FileNotFoundError(f"Cannot find files in directory: {input_dir}.")

    diff = set(ids) - set(avail_ids.keys())
    if len(diff):
        raise ValueError(f'IDs in id_file not found in input_dir: {diff}')

    # Create a tempdir
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    # Create link of files
    for id in ids:
        logger.debug(f"Creating link for {id}")
        src = str(avail_ids[id][0].absolute())
        dst = temp_path / avail_ids[id][0].name
        (temp_path / avail_ids[id][0].name).symlink_to(src)
    return temp_dir


def run_inference_pipeline(
        cfg: Dict,
        debug: bool,
        id_file: Optional[Path],
        id_globber: Optional[str],
        id_list: Optional[List[str]],
        inference: bool,
        inference_resample_to_origin: bool,
        input_dir: Path,
        keep_intermediate_segments: bool,
        main_logger: MNTSLogger,
        models_dir: Path,
        output_dir: Path,
        sequence: str,
        skip_norm: bool,
        skip_post_processing: bool,
        t_start: float,
        resume: bool = False,
        cleanup_progress: bool = False,
        input_probmap_dir: Optional[Path] = None
    ) -> None:
    """Run the complete NPC segmentation pipeline with progress persistence support.

    This function executes the full pipeline for NPC (Nasopharyngeal Carcinoma) segmentation:
    1. Normalize the input images (optional)
    2. Perform coarse segmentation
    3. Grow the coarse segmentation
    4. Perform fine segmentation
    5. Post-process the results (optional)
    6. Resample to original space (optional)

    Args:
        cfg (DotDict):
            Configuration object containing pipeline settings
        debug (bool):
            Whether to run in debug mode with limited samples
        id_file (Optional[Path]):
            Path to file containing list of case IDs
        id_globber (Optional[str]):
            Glob pattern to match files with specific IDs
        id_list (Optional[List[str]]):
            List of case IDs to process
        inference (bool):
            Whether to perform inference
        inference_resample_to_origin (bool):
            Whether to resample results back to original space
        input_dir (Path):
            Directory containing input images
        keep_intermediate_segments (bool):
            Whether to save intermediate segmentation results
        main_logger (Logger):
            Logger object for pipeline logging
        output_dir (Path):
            Directory to save final results
        sequence (str):
            MRI sequence type (e.g., 'T1W', 'T2W')
        skip_norm (bool):
            Whether to skip normalization step
        skip_post_processing (bool):
            Whether to skip post-processing step
        t_start (float):
            Pipeline start time for timing tracking
        resume (bool):
            Whether to resume from previous interrupted run
        cleanup_progress (bool):
            Whether to clean up progress files at the end
    Raises:
        ValueError: If id_globber is not set when using id_file or id_list
        FileNotFoundError: If input directory structure is incorrect for skipped normalization

    .. note::
        The function creates temporary directories for intermediate results and
        cleans them up after completion. If keep_intermediate_segments is True,
        intermediate results are copied to subdirectories in the output_dir.
        Progress persistence is supported to enable recovery from interruptions.
    """
    
    # Initialize progress manager
    pipeline_config = {
        'sequence': sequence,
        'input_dir': str(input_dir),
        'id_globber': id_globber,
        'skip_norm': skip_norm,
        'skip_post_processing': skip_post_processing,
        'inference_resample_to_origin': inference_resample_to_origin
    }
    
    progress_manager = ProgressStateManager(
        output_dir=output_dir,
        config=pipeline_config,
        logger=main_logger
    )
    
    # Handle cleanup request
    if cleanup_progress:
        progress_manager.cleanup_progress()
        main_logger.info("Progress files cleaned up")
    else:
        # Load or create progress state
        progress_manager.load_or_create_state()
    
    # Check for resume
    if resume:
        resume_step = progress_manager.get_resume_point()
        if resume_step != 'input_prepared':
            main_logger.info(f"Resuming from step: {resume_step}")
        else:
            main_logger.info("No previous progress found, starting from beginning")
    # * Setup
    # Override sequence options
    cfg.sequence = sequence

    # Package assets: YAML configs bundled with the package
    assets_path = Path(__file__).parent.absolute() / 'assets'
    normalization_graph = assets_path / "normalization_t2w.yaml"
    # Model assets: checkpoints and normalizer states, mounted externally
    models_dir_path = Path(models_dir)
    normalization_states = models_dir_path / f"norm_states/{sequence}"
    main_logger.info(f"{assets_path = }")
    main_logger.info(f"{models_dir_path = }")
    main_logger.info(f"{normalization_graph = }")
    main_logger.info(f"{normalization_states = }")
    # pass the settings to cfg
    cfg.models_dir = str(models_dir_path)
    if id_globber is not None:
        cfg.data_loader_cfg.id_globber = id_globber
    
    # Check if input preparation is needed
    if not progress_manager.is_step_completed('input_prepared'):
        progress_manager.mark_step_started('input_prepared')
        # Input preparation logic will be handled below
        main_logger.info("Preparing inputs...")
    else:
        main_logger.info("Input prepared, skipping this step.")
    
    # * Perform normalization of inputs
    with (tempfile.TemporaryDirectory() as normed_tempdir, \
          tempfile.TemporaryDirectory() as output_tempdir):
        # Preserve original input dir to resolve probmap dir correctly
        orig_input_dir = input_dir
        source_probmap_dir = input_probmap_dir or orig_input_dir / "HuangThresholding"
        probmap_temp_dir = None

        # If id file or id list is specified, create another temp dir to hold the links to inputs
        # Note that in this mode we still assume Huang's Thresholding and Nyulnormalizer folder
        if id_file is not None or id_list is not None:
            if id_globber is None:
                raise ValueError("id_globber must be set to use id_file!")
            main_logger.info("Creating temp dir for holding specified id lists...")
            temp_dir = create_temp_dir_for_inference(id_file or id_list, id_globber, input_dir)
            input_dir = Path(temp_dir.name)

            if source_probmap_dir.is_dir():
                main_logger.info("Creating temp dir for prob map...")
                probmap_temp_dir = create_temp_dir_for_inference(id_list, id_globber, source_probmap_dir)

        # Update probmap_dir if it was successfully filtered into a temp dir
        if probmap_temp_dir is not None:
            probmap_dir = Path(probmap_temp_dir.name)
        else:
            probmap_dir = source_probmap_dir

        # Mark input prepared as completed
        if not progress_manager.is_step_completed('input_prepared'):
            progress_manager.mark_step_completed('input_prepared')
        
        # Specify where to place normalized images
        normed_tempdir_path = Path(normed_tempdir)

        # Check normalization step
        if not skip_norm and not progress_manager.is_step_completed('normalization_completed'):
            progress_manager.mark_step_started('normalization_completed')
            main_logger.info("Start normalization...")
            # Create normalizer
            norm = NPCSegmentPreprocesser(normalization_graph, state_dir=normalization_states)
            norm.input_dir = input_dir
            norm.output_dir = normed_tempdir_path
            # debug
            if debug:
                main_logger.info("Running in debug mode...")
                debug_path = tempfile.TemporaryDirectory()
                # copy 3 images to this path from the original input
                for i, r in enumerate(Path(input_dir).rglob('*nii.gz')):
                    if i == 2:
                        break
                    shutil.copy2(r, debug_path.name)
                norm.input_dir = debug_path.name

            norm.exec()
            if debug:
                debug_path.cleanup()

            # Copy normalized output if requested and mark step completed
            if keep_intermediate_segments:
                shutil.copytree(normed_tempdir_path, output_dir / "Norm_output", dirs_exist_ok=True)
            
            # Mark normalization completed and stream output
            progress_manager.mark_step_completed('normalization_completed', normed_tempdir_path)
            
        elif not skip_norm:
            # Continuing run but normalization was already done
            main_logger.info("Normalization already done. Skipping.")
        else:
            # If skip norm, would still need to obtain the probability maps
            main_logger.info("Skipping normalization step.")
            main_logger.info("Creating symlink to temp work directory...")

            # replace the original temp dir with a new symlink dir to hold normalization results
            normed_tempdir_path = normed_tempdir_path / "Symlinkdir"
            normed_tempdir_path.mkdir()
            nyul_path = normed_tempdir_path / "NyulNormalizer" # We still assume there's such a folder

            # This mean we are using temp dir to hold those that are part of the specified IDs
            if id_file is not None or id_list is not None:
                # Create the folder 'NyulNormalizer' in temp directory and move all symlinks in.
                nyul_path.mkdir(parents=True, exist_ok=True)
                main_logger.info("Recreating symlinks to isolate files...")
                main_logger.info(f"{input_dir = }")
                link_dst = nyul_path
                symlink_to_move = input_dir.glob('*.nii.gz')
                for f in symlink_to_move:
                    main_logger.info(f"Linking {f} to {nyul_path}")
                    link_dst.joinpath(f.name).symlink_to(f.resolve())
            else:
                # Directly link to the specified input dir to deal with all files
                main_logger.info(f"Linking: {nyul_path.absolute()} -> {input_dir.absolute()}")
                nyul_path.symlink_to(input_dir.absolute(), target_is_directory=True)

            if not probmap_dir.is_dir():
                # We need probmap for sampling the patches
                main_logger.warning(f"Cannot find HuanThresholding files, trying to create it.")
                probmap_dir.mkdir()

                for f in nyul_path.glob("*.nii.gz"):
                    main_logger.info(f"Performing thresholding on {f}")
                    _sitk_im = sitk.ReadImage(str(f))
                    # Because the input is expected to be normalized, we can use Otsu thresholding at 0
                    _tissue_mask = _sitk_im > 0  #
                    main_logger.info(f"Saving tissue mask to {str(probmap_dir / f.name)}")
                    sitk.WriteImage(_tissue_mask, str(probmap_dir / f.name))

            # link the probmap dir
            probmap_link_path = normed_tempdir_path / "HuangThresholding"
            probmap_link_path.symlink_to(probmap_dir.absolute(), target_is_directory=True)


        # * Coarse segmentation
        output_tempdir_path = Path(output_tempdir)
        coarse_out_path = output_tempdir_path / "Coarse"
        
        if not progress_manager.is_step_completed('coarse_segmentation_completed'):
            progress_manager.mark_step_started('coarse_segmentation_completed')
            main_logger.info("Start Segmentation...")

            run_inference(cfg,
                          normed_tempdir_path / "NyulNormalizer",
                          coarse_out_path,
                          normed_tempdir_path / "HuangThresholding")

            # Copy coarse output if requested
            if keep_intermediate_segments:
                shutil.copytree(coarse_out_path, output_dir / "Coarse_output", dirs_exist_ok=True)
            
            # Mark coarse segmentation completed and stream output
            progress_manager.mark_step_completed('coarse_segmentation_completed', coarse_out_path)
        else:
            main_logger.info("Segmentation already done. Skipping.")
            # If skipping, we still need the coarse output for next steps
            # Try to restore from intermediate directory
            if (progress_manager.intermediate_dir / "Coarse_output").exists():
                shutil.copytree(progress_manager.intermediate_dir / "Coarse_output", coarse_out_path, dirs_exist_ok=True)

        # Growth step
        if not progress_manager.is_step_completed('segmentation_growth_completed'):
            progress_manager.mark_step_started('segmentation_growth_completed')
            
            # grow the segmentation a bit for better capturing the NPC
            grow_segmentation(str(coarse_out_path))

            # Copy grow_segmentation if requested
            if keep_intermediate_segments:
                shutil.copytree(coarse_out_path, output_dir / "Growth_output", dirs_exist_ok=True)
            
            progress_manager.mark_step_completed('segmentation_growth_completed', coarse_out_path)
        else:
            pass

        # * Fine segmentation
        fine_out_path = output_tempdir_path / "Fine"
        
        if not progress_manager.is_step_completed('fine_segmentation_completed'):
            progress_manager.mark_step_started('fine_segmentation_completed')
            main_logger.info("Start refined segmentation...")

            # Check if any of the corase segmentation is empty
            for f in coarse_out_path.glob("*nii.gz"):
                # Use SimpleITK to compute volume (in milliliters) for each coarse segmentation
                im = sitk.ReadImage(str(f))
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(sitk.Cast(im > 0, sitk.sitkUInt8))
                is_empty = not stats.HasLabel(1) or stats.GetNumberOfPixels(1) == 0
                if is_empty:
                    main_logger.warning(f"{f.name} is empty after coarse segmentation.")

            # Update the directories and redo the segmentation with coarse segment as probmap
            run_inference(cfg,
                          normed_tempdir_path / "NyulNormalizer",
                          fine_out_path,
                          coarse_out_path)

            # Copy fine segmentation if requested
            if keep_intermediate_segments:
                shutil.copytree(fine_out_path, output_dir / "Finesegment_output", dirs_exist_ok=True)
            
            progress_manager.mark_step_completed('fine_segmentation_completed', fine_out_path)
        else:
            main_logger.info("Refined segmentaiton done. Skipping")
            # Restore from intermediate directory if needed
            if (progress_manager.intermediate_dir / "Fine_output").exists():
                shutil.copytree(progress_manager.intermediate_dir / "Fine_output", fine_out_path, dirs_exist_ok=True)

        # * Post-processing
        pp_out_path = output_tempdir_path / "PostProcessed"
        pp_out_path.mkdir(exist_ok=True)

        if not progress_manager.is_step_completed('post_processing_completed'):
            progress_manager.mark_step_started('post_processing_completed')
            main_logger.info("Start post-processing...")
            
            # Option to skip this
            if not skip_post_processing:
                seg_post_main(fine_out_path, pp_out_path)
            else:
                # Note that if intermediate output is kept, a duplicate of fine segment will be generated as output
                shutil.copytree(fine_out_path, pp_out_path, dirs_exist_ok=True)

            progress_manager.mark_step_completed('post_processing_completed', pp_out_path)
        else:
            main_logger.info("Post-processing done. Skipping")
            # Restore from intermediate directory if needed
            if (progress_manager.intermediate_dir / "PostProcessed_output").exists():
                shutil.copytree(progress_manager.intermediate_dir / "PostProcessed_output", pp_out_path, dirs_exist_ok=True)

        # * Resample to original space if specified
        if not progress_manager.is_step_completed('resampling_completed'):
            if inference_resample_to_origin and inference:
                progress_manager.mark_step_started('resampling_completed')
                main_logger.info("Resmapling to original space...")
                resample_pairs(pp_out_path, input_dir, pp_out_path)
            
            progress_manager.mark_step_completed('resampling_completed', pp_out_path)
        else:
            pass

        # * Move results to output path
        if not progress_manager.is_step_completed('output_completed'):
            progress_manager.mark_step_started('output_completed')
            main_logger.info("Copying output...")
            
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True, exist_ok=True)

            for f in pp_out_path.rglob("*nii.gz"):
                main_logger.info(f"Copying: {str(f)} -> {str(output_dir.absolute())}")
                shutil.copy2(str(f.absolute()), str(output_dir.absolute()))
            
            progress_manager.mark_step_completed('output_completed')
        else:
            pass

        # Clean the temporary directory
        if id_file is not None or id_list is not None:
            temp_dir.cleanup()
            if probmap_temp_dir is not None:
                probmap_temp_dir.cleanup()

    # Generate final progress report
    progress_report = progress_manager.generate_progress_report()
    main_logger.info("Progress Report:")
    main_logger.info(progress_report)
    
    # Save final state
    progress_manager.save_state()
    
    # Export progress log if requested
    progress_log_file = output_dir / 'progress_report.txt'
    progress_manager.export_progress_log(progress_log_file)
    
    # Cleanup progress files if requested
    if cleanup_progress:
        progress_manager.cleanup_progress()
        main_logger.info("Progress files cleaned up")
    
    main_logger.info("{:=^80}".format(f" Segmentation Done (Total: {time.time() - t_start:.01f}s) "))


def run_inference(cfg: PMIControllerCFG,
                  input_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  probmap_dir: Union[str, Path]) -> None:
    """Runs the inference process for NPC segmentation.

    This function overrides the input, output, and probability map directories specified
    in the configuration object with the provided directory paths, creates a
    PMIController with the updated configuration, and then executes the inference
    process. After execution, the PMIController instance is deleted.

    Args:
        cfg (NPCSegmentControllerCFG):
            The configuration object for NPC segmentation. This object will be copied and
            its directories will be overridden with the provided paths.
        input_dir (str):
            The directory path where input data is located.
        output_dir (str):
            The directory path where output data should be saved.
        probmap_dir (str):
            The directory path where probability maps are located.

    .. note::
            This function does not return any value.

    """
    # Override directories to a temp directory
    cfg: NPCSegmentControllerCFG = copy.copy(cfg)
    cfg.data_loader_cfg.input_dir = input_dir
    cfg.data_loader_cfg.probmap_dir = probmap_dir
    cfg.output_dir = output_dir # Note that output directory is controller's property because dataloader is only for
                                # loading data.

    # Create controller
    controller = PMIController(cfg)
    # Execute
    controller.exec()
    del controller


def run_training(cfg: PMIControllerCFG,
                  input_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  probmap_dir: Union[str, Path]) -> None:
    """Runs the training process for NPC segmentation.

    """
    # Override directories to a temp directory
    cfg: NPCSegmentControllerCFG = copy.copy(cfg)
    cfg.data_loader_cfg.input_dir = input_dir
    cfg.data_loader_cfg.probmap_dir = probmap_dir
    cfg.output_dir = output_dir # Note that output directory is controller's property because dataloader is only for
                                # loading data.

    # Create controller
    controller = PMIController(cfg)

    # Execute
    controller.exec()
    del controller

@click.command()
@click.argument('input-dir', required=True, type=click.Path())
@click.argument('output-dir', required=True, type=click.Path())
@click.option('--models-dir',
              default=lambda: os.environ.get('NPC_MODELS_DIR', '/models'),
              show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Directory containing model checkpoints and normalization states. "
                   "Defaults to $NPC_MODELS_DIR env var or /models.")
@click.option('--input-probmap-dir', default=None, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Specify the path to the probability map. If not given and not exist in `input-dir/HuangsThreshold` "
                   "folder, we will create one for you but doesn't gaurunteer it works. This option only works in "
                   "inference mode")
@click.option('--id-list', default=None, type=str,
              help="CSV string of IDs to inference on testing set, or a file path to inference on all ids. Ignored if "
                   "--id-file is not None or `--id-globber` is None.")
@click.option('--id-file', default=None, type=click.Path(),
              help='Specify a .ini file to inference on testing set, or a .txt file to inference on all ids. The IDs'
                   'are globbed by argument --id-globber.')
@click.option('--id-globber', default=None, type=str,
              help="Regex pattern to glob IDs from input directory. Ignored if --id-ini is not None.")
@click.option('--sequence', default='T2WFS', nargs=1,
              type=click.Choice(sequence_choice, case_sensitive=True),
              help=f"Set the sequence. Chose from [{','.join(sequence_choice)}]")
@click.option('--skip-norm', default=False, is_flag=True,
              help="If true, skip intensity normalization")
@click.option('--skip-post-process', is_flag=True,
              help='When specified, the post-processing step will be skipped.')
@click.option('--debug', default=False, is_flag=True,
              help="If true, only operate on the first three case globbed.")
@click.option('--inference', default=False, is_flag=True,
              help="For guild operation")
@click.option("--keep-intermediate-segments", default=False, is_flag=True,
              help="If specified, keep intermediate data created.")
@click.option('--inference-resample-to-origin', is_flag=True,
              help="If specified, the inference output will be resampled to the input space.")
@click.option('--debug', default=False, is_flag=True,
              help="If true, only operate on the first three cases globbed.")
@click.option('--resume', default=False, is_flag=True,
              help="Resume from previous interrupted run using progress state.")
@click.option('--cleanup-progress', default=False, is_flag=True,
              help="Clean up progress files and start fresh.")
def main(input_dir : PathLike,
         output_dir: PathLike,
         models_dir: PathLike,
         input_probmap_dir: PathLike,
         id_list: str,
         id_file: PathLike,
         id_globber: str,
         sequence  : str,
         inference : bool,
         skip_norm : Optional[bool] = False,
         skip_post_processing: Optional[bool] = False,
         debug     : Optional[bool] = False,
         keep_intermediate_segments: Optional[bool] = False,
         inference_resample_to_origin: Optional[bool] = False,
         resume: Optional[bool] = False,
         cleanup_progress: Optional[bool] = False,
         **kwargs):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    main_logger = MNTSLogger('.', 'main', verbose=True, keep_file=False, log_level='debug')
    t_start = time.time()
    main_logger.info("{:=^80}".format(" NPC auto Segmentation Running "))

    # Create controller
    cfg = NPCSegmentControllerCFG()
    cfg.sequence = sequence

    # Guild is giving us trouble for double printing everything
    cfg.verbose = False
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    if inference:
        main_logger.info("Running inference...")
        cfg.run_mode = 'inference'
        run_inference_pipeline(cfg, debug, id_file, id_globber, id_list, inference, inference_resample_to_origin,
                               input_dir, keep_intermediate_segments, main_logger, models_dir, output_dir, sequence,
                               skip_norm, skip_post_processing, t_start, resume, cleanup_progress, input_probmap_dir)
    else:
        raise NotImplementedError("Training is not available yet!")
        logger.info("Running training...")
        if not skip_norm:
            logger.error("Currently don't support training")

        # Set directories
        cfg.data_loader_cfg.input_dir = input_dir
        cfg.output_dir = output_dir
        cfg.run_mode = 'training'



if __name__ == '__main__':
    main()
