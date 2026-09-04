import re
import click
import mnts
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.dicom2nii import dicom2nii
from mnts.filters.mpi_wrapper import mpi_wrapper, repeat_zip
from pathlib import Path


@click.command(help='CLI for normalizing MRI images using the MNTS package.')
@click.option('--input-dir', '-i', type=click.Path(exists=True, dir_okay=True, path_type=Path, required=True),
              help='Directory where input data is located.')
@click.option('--output-dir', '-o',
              type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=Path), required = True,
              help='Directory where output data will be saved.')
@click.option('--mnts-state', '-s',
              type=click.Path(dir_okay=True, file_okay=True, readable=True, path_type=Path),
              help='Path to the MNTS state file required for normalization steps that utilize training.')
@click.option('--mnts-graph', '-g',
              type=click.Path(file_okay=True, readable=True, path_type=Path), required = True,
              help='Path to the MNTS graph file needed for normalization steps that involve training.')
@click.option('--num-workers', '-n', type=int, help='Number of workers to use.')
@click.option('--keep-log', is_flag=True,
              help='Flag to keep logs. Default log directory is ./[Date]-Normalization.log')
@click.option('--log-level', default='INFO', help='Logging level.')
@click.option('--verbose', '-v', is_flag=True, help="Output to STDOUT.")
def normalize_input(input_dir   : Path,
                    output_dir : Path,
                    mnts_state : Path,
                    mnts_graph : Path,
                    num_workers: int,
                    keep_log   : bool,
                    log_level  : str,
                    verbose    : bool):
    # create logger
    if keep_log:
        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Create the log file name
        log_file_name = f"{current_date}-Normalization.log"
    with MNTSLogger(log_file_name=log_file_name, keep_file=keep_log, log_level=log_leve, verbose=verbose) as logger:
        logger.info("{:=^150}".format(" Initiating Normalization "))

        # creates the graph
        G = mnts.filters.MNTSFilterGraph()
        G.CreateGraphFromYAML(mnts_graph)

        # load states
        if G.requires_training:
            G.load_node_states(None, mnts_state)

        # * MPI normalization
        # First gather all input data
        input_fpaths = list(input_dir.rglob('*.nii.gz'))
        # Decide output directories
        output_prefix = [f.name for f in input_fpaths] # name of output files
        input_vector = [output_prefix, [output_dir], input_fpaths]

        if num_workers >= 1:
            logger.info("Start processing input without parallelization.")
            for row in repeat_zip(*input_vector):
                G.mpi_execute(*row)
        else:
            logger.info(f"Start processing input with {num_workers} workers.")
            mpi_wrapper(G.mpi_execute, input_vector, num_worker=num_workers)

if __name__ == '__main__':
    normalize_input()