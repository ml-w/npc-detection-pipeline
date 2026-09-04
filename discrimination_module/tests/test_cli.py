import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner
import yaml

from rAIdiologist.cli.cli_normalize_input import normalize_input, DEFAULT_PARAMS


class TestCliNormalizeInput(unittest.TestCase):
    """Test suite for cli_normalize_input module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create test input files
        self.test_files = [
            self.input_dir / "subject_001.nii.gz",
            self.input_dir / "subject_002.nii.gz",
            self.input_dir / "subject_003.nii.gz"
        ]
        for f in self.test_files:
            f.touch()
        
        # Create mock MNTS graph file
        self.graph_file = self.test_dir / "test_graph.yaml"
        test_graph_content = {
            'SpatialNorm': {
                'out_spacing': [0.4492, 0.4492, 4]
            },
            'HuangThresholding': {
                'closing_kernel_size': 10,
                '_ext': {
                    'upstream': 0,
                    'is_exit': True
                }
            },
            'N4ITKBiasFieldCorrection': {
                '_ext': {
                    'upstream': [0, 1]
                }
            },
            'NyulNormalizer': {
                '_ext': {
                    'upstream': [2, 1],
                    'is_exit': True
                }
            }
        }
        with open(self.graph_file, 'w') as f:
            yaml.dump(test_graph_content, f)
        
        # Create mock state directory
        self.state_dir = self.test_dir / "state"
        self.state_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.test_dir)

    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_basic_functionality(self, mock_logger, mock_create_graph):
        """Test basic normalization functionality without parallelization."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_graph.requires_training = False
        mock_graph.mpi_execute = MagicMock()
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        # Run the command
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file),
            '--num-workers', '1'
        ])
        
        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_create_graph.assert_called_once_with(self.graph_file)
        mock_graph.set_progress_bar.assert_called_once()
        mock_graph.close_progress_bar.assert_called_once()
        self.assertEqual(mock_graph.mpi_execute.call_count, len(self.test_files))

    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_with_training_state(self, mock_logger, mock_create_graph):
        """Test normalization with training state loading."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_graph.requires_training = True
        mock_graph.load_node_states = MagicMock()
        mock_graph.mpi_execute = MagicMock()
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        # Run the command with state
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file),
            '--mnts-state', str(self.state_dir),
            '--num-workers', '1'
        ])
        
        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_graph.load_node_states.assert_called_once_with(None, self.state_dir)

    @patch('rAIdiologist.cli.cli_normalize_input.mpi_wrapper')
    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_parallel_processing(self, mock_logger, mock_create_graph, mock_mpi_wrapper):
        """Test normalization with parallel processing."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_graph.requires_training = False
        mock_graph.mpi_execute = MagicMock()
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        # Run the command with multiple workers
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file),
            '--num-workers', '4'
        ])
        
        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_mpi_wrapper.assert_called_once()
        # Verify that mpi_wrapper was called with correct parameters
        args, kwargs = mock_mpi_wrapper.call_args
        self.assertEqual(args[0], mock_graph.mpi_execute)
        self.assertEqual(kwargs['num_worker'], 4)

    @patch('mnts.utils.filename_globber.get_unique_IDs')
    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_id_verification_success(self, mock_logger, mock_create_graph, mock_get_unique_ids):
        """Test ID verification when all IDs match."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_graph.requires_training = False
        mock_graph.mpi_execute = MagicMock()
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        # Mock successful ID verification
        mock_get_unique_ids.return_value = ['001', '002', '003']
        
        # Create output files for verification
        for i, filename in enumerate(['subject_001.nii.gz', 'subject_002.nii.gz', 'subject_003.nii.gz']):
            (self.output_dir / filename).touch()
        
        # Run the command with ID verification
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file),
            '--id-globber', r'subject_(\d+)',
            '--num-workers', '1'
        ])
        
        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_logger_instance.warning.assert_not_called()

    @patch('mnts.utils.filename_globber.get_unique_IDs')
    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_id_verification_missing_ids(self, mock_logger, mock_create_graph, mock_get_unique_ids):
        """Test ID verification when some IDs are missing."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_graph.requires_training = False
        mock_graph.mpi_execute = MagicMock()
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        # Mock missing ID verification
        def mock_get_ids_side_effect(filenames, pattern):
            if 'subject_001.nii.gz' in filenames:  # input files
                return ['001', '002', '003']
            else:  # output files
                return ['001', '002']  # missing '003'
        
        mock_get_unique_ids.side_effect = mock_get_ids_side_effect
        
        # Create partial output files
        (self.output_dir / 'subject_001.nii.gz').touch()
        (self.output_dir / 'subject_002.nii.gz').touch()
        
        # Run the command with ID verification
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file),
            '--id-globber', r'subject_(\d+)',
            '--num-workers', '1',
            '--verbose'
        ])
        
        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_logger_instance.warning.assert_called_once()
        warning_call = mock_logger_instance.warning.call_args[0][0]
        self.assertIn('003', warning_call)

    def test_missing_input_directory(self):
        """Test error handling when input directory doesn't exist."""
        non_existent_dir = self.test_dir / "non_existent"
        
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(non_existent_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file)
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('does not exist', result.output)

    def test_missing_graph_file(self):
        """Test error handling when graph file doesn't exist."""
        non_existent_graph = self.test_dir / "non_existent.yaml"
        
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(self.input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(non_existent_graph)
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Cannot open transform file', result.output)

    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_logging_options(self, mock_logger):
        """Test various logging options."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        with patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML') as mock_create_graph:
            mock_graph = MagicMock()
            mock_graph.requires_training = False
            mock_create_graph.return_value = mock_graph
            
            # Test with keep-log and verbose flags
            result = self.runner.invoke(normalize_input, [
                '--input-dir', str(self.input_dir),
                '--output-dir', str(self.output_dir),
                '--mnts-graph', str(self.graph_file),
                '--keep-log',
                '--verbose',
                '--log-level', 'DEBUG'
            ])
            
            # Verify logger was called with correct parameters
            mock_logger.assert_called_once()
            args, kwargs = mock_logger.call_args
            self.assertTrue(kwargs['keep_file'])
            self.assertTrue(kwargs['verbose'])
            self.assertEqual(kwargs['log_level'], 'DEBUG')

    def test_default_parameters(self):
        """Test that default parameters are correctly set."""
        self.assertTrue(DEFAULT_PARAMS['mnts_state'].exists(), f"{DEFAULT_PARAMS['mnts_state']} does not exist")
        self.assertTrue(DEFAULT_PARAMS['mnts_graph'].exists(), f"{DEFAULT_PARAMS['mnts_graph']} does not exist")

    @patch('rAIdiologist.cli.cli_normalize_input.mnts.filters.MNTSFilterGraph.CreateGraphFromYAML')
    @patch('rAIdiologist.cli.cli_normalize_input.MNTSLogger')
    def test_no_input_files(self, mock_logger, mock_create_graph):
        """Test behavior when no .nii.gz files are found in input directory."""
        # Create empty input directory
        empty_input_dir = self.test_dir / "empty_input"
        empty_input_dir.mkdir()
        
        mock_graph = MagicMock()
        mock_graph.requires_training = False
        mock_create_graph.return_value = mock_graph
        
        mock_logger_instance = MagicMock()
        mock_logger.return_value.__enter__.return_value = mock_logger_instance
        
        result = self.runner.invoke(normalize_input, [
            '--input-dir', str(empty_input_dir),
            '--output-dir', str(self.output_dir),
            '--mnts-graph', str(self.graph_file)
        ])
        
        # Should fail directly because click checks for valid input-dir
        self.assertEqual(result.exit_code, 1)


if __name__ == '__main__':
    unittest.main()