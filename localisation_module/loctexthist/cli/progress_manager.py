"""
Progress State Manager Module

This module provides progress tracking, persistence, and recovery functionality
for the NPC segmentation pipeline. Supports fine-grained recovery control at
both step level and subject level.
"""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

from mnts.utils import get_unique_IDs

PathLike = Union[str, Path]


class ProgressStateError(Exception):
    """Progress state related errors"""
    pass


class ProgressValidationError(ProgressStateError):
    """Progress validation errors"""
    pass


class ProgressCorruptionError(ProgressStateError):
    """Progress corruption errors"""
    pass


class ProgressStateManager:
    """Manages progress state and recovery mechanisms for NPC segmentation pipeline
    
    This class is responsible for:
    1. Tracking and persisting pipeline execution progress
    2. Supporting recovery from any checkpoint
    3. Providing fine-grained control at subject level
    4. Implementing real-time streaming output functionality
    """
    
    # Pipeline step definitions
    PIPELINE_STEPS = [
        'input_prepared',
        'normalization_completed', 
        'coarse_segmentation_completed',
        'segmentation_growth_completed',
        'fine_segmentation_completed',
        'post_processing_completed',
        'resampling_completed',
        'output_completed'
    ]
    
    # Steps that require subject-level tracking
    SUBJECT_LEVEL_STEPS = [
        'coarse_segmentation_completed',
        'fine_segmentation_completed'
    ]
    
    # Step corresponding output directory names
    STEP_OUTPUT_DIRS = {
        'normalization_completed': 'Norm_output',
        'coarse_segmentation_completed': 'Coarse_output',
        'fine_segmentation_completed': 'Fine_output',
        'post_processing_completed': 'PostProcessed_output'
    }
    
    # State file name
    STATE_FILE = '.npc_progress.json'
    
    def __init__(self, output_dir: PathLike, config: Dict, logger: Optional[logging.Logger] = None):
        """Initialize progress manager
        
        Args:
            output_dir: Final output directory
            config: Pipeline configuration dictionary
            logger: Logger instance, will create default logger if not provided
        """
        self.output_dir = Path(output_dir)
        self.config = config.copy()
        self.logger = logger or self._create_default_logger()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State file path
        self.state_file_path = self.output_dir / self.STATE_FILE
        
        # Intermediate output directory
        self.intermediate_dir = self.output_dir / 'intermediate'
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Load or create state
        self.state = self.load_or_create_state()
    
    def _create_default_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger('npc_progress')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_or_create_state(self) -> Dict:
        """Load existing state or create new state
        
        Returns:
            State dictionary
        """
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Validate state file integrity
                self._validate_state_structure(state)
                
                # Check if configuration matches (important parameters)
                if not self._is_config_compatible(state.get('configuration', {})):
                    self.logger.warning("Configuration has changed, creating new progress state")
                    return self._create_new_state()
                
                self.logger.info(f"Loaded existing progress state, pipeline ID: {state['pipeline_id']}")
                return state
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error(f"State file corrupted: {e}")
                self.logger.info("Creating new progress state")
                return self._create_new_state()
        else:
            return self._create_new_state()
    
    def _create_new_state(self) -> Dict:
        """Create new state dictionary"""
        pipeline_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Get subject ID list
        subject_ids = self._extract_subject_ids()
        
        state = {
            'pipeline_id': pipeline_id,
            'start_time': current_time,
            'last_updated': current_time,
            'configuration': self.config,
            'input_info': {
                'total_subjects': len(subject_ids),
                'subject_ids': subject_ids
            },
            'step_status': {step: False for step in self.PIPELINE_STEPS},
            'subject_progress': {step: [] for step in self.SUBJECT_LEVEL_STEPS},
            'output_streaming': {
                'intermediate_dir': str(self.intermediate_dir),
                'streamed_steps': []
            }
        }
        
        self.logger.info(f"Created new progress state, pipeline ID: {pipeline_id}")
        self.logger.info(f"Found {len(subject_ids)} subjects: {subject_ids}")
        
        return state
    
    def _extract_subject_ids(self) -> List[str]:
        """Extract subject ID list from input directory
        
        Returns:
            Subject ID list
        """
        input_dir = Path(self.config.get('input_dir', ''))
        id_globber = self.config.get('id_globber', "^[a-zA-Z]{0,5}[0-9]+")
        
        if not input_dir.exists():
            self.logger.warning(f"Input directory does not exist: {input_dir}")
            return []
        
        try:
            # Use MNTS utility function to extract IDs
            nii_files = list(input_dir.rglob('*.nii.gz'))
            if not nii_files:
                nii_files = list(input_dir.rglob('*.nii'))
            
            if nii_files:
                ids_dict = get_unique_IDs(nii_files, id_globber, return_dict=True)
                subject_ids = list(ids_dict.keys())
                subject_ids.sort()
                return subject_ids
            else:
                self.logger.warning("No NII files found")
                return []
                
        except Exception as e:
            self.logger.error(f"Error occurred while extracting subject IDs: {e}")
            return []
    
    def _validate_state_structure(self, state: Dict):
        """Validate state file structure integrity
        
        Args:
            state: State dictionary
            
        Raises:
            ProgressValidationError: When state structure is invalid
        """
        required_keys = [
            'pipeline_id', 'start_time', 'last_updated', 'configuration',
            'input_info', 'step_status', 'subject_progress', 'output_streaming'
        ]
        
        for key in required_keys:
            if key not in state:
                raise ProgressValidationError(f"State file missing required field: {key}")
        
        # Check step status
        for step in self.PIPELINE_STEPS:
            if step not in state['step_status']:
                raise ProgressValidationError(f"Step status missing: {step}")
        
        # Check subject progress
        for step in self.SUBJECT_LEVEL_STEPS:
            if step not in state['subject_progress']:
                raise ProgressValidationError(f"Subject progress missing: {step}")
    
    def _is_config_compatible(self, saved_config: Dict) -> bool:
        """Check compatibility between current configuration and saved configuration
        
        Args:
            saved_config: Saved configuration
            
        Returns:
            Whether compatible
        """
        # Check key configuration parameters
        key_params = ['sequence', 'input_dir', 'id_globber']
        
        for param in key_params:
            if self.config.get(param) != saved_config.get(param):
                self.logger.warning(f"Configuration parameter {param} changed: {saved_config.get(param)} -> {self.config.get(param)}")
                return False
        
        return True
    
    def save_state(self):
        """Atomically save state to file"""
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Use temporary file to ensure atomic write
        temp_file = self.state_file_path.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.state_file_path)
            
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise ProgressStateError(f"Failed to save state file: {e}")
    
    def is_step_completed(self, step: str) -> bool:
        """Check if step is completed
        
        Args:
            step: Step name
            
        Returns:
            Whether completed
        """
        return self.state['step_status'].get(step, False)
    
    def is_subject_completed(self, step: str, subject_id: str) -> bool:
        """Check if step is completed for specific subject
        
        Args:
            step: Step name
            subject_id: Subject ID
            
        Returns:
            Whether completed
        """
        if step not in self.SUBJECT_LEVEL_STEPS:
            return self.is_step_completed(step)
        
        completed_subjects = self.state['subject_progress'].get(step, [])
        return subject_id in completed_subjects
    
    def get_remaining_subjects(self, step: str) -> List[str]:
        """Get list of subjects that haven't completed the specified step
        
        Args:
            step: Step name
            
        Returns:
            List of incomplete subject IDs
        """
        if step not in self.SUBJECT_LEVEL_STEPS:
            return []
        
        all_subjects = self.state['input_info']['subject_ids']
        completed_subjects = self.state['subject_progress'].get(step, [])
        
        remaining = [sid for sid in all_subjects if sid not in completed_subjects]
        return remaining
    
    def get_resume_point(self) -> Tuple[str, List[str]]:
        """Get resume point and subjects that need processing
        
        Returns:
            (Next step to execute, List of subjects to process)
        """
        # Find first incomplete step
        for step in self.PIPELINE_STEPS:
            if not self.is_step_completed(step):
                if step in self.SUBJECT_LEVEL_STEPS:
                    remaining_subjects = self.get_remaining_subjects(step)
                    return step, remaining_subjects
                else:
                    return step, []
        
        # All steps completed
        return 'output_completed', []
    
    def mark_step_started(self, step: str):
        """Mark step as started
        
        Args:
            step: Step name
        """
        self.logger.info(f"Starting step: {step}")
        self.save_state()
    
    def mark_step_completed(self, step: str, temp_output_dir: Optional[Path] = None):
        """Mark step as completed and trigger real-time streaming
        
        Args:
            step: Completed step name
            temp_output_dir: Temporary output directory (for streaming)
        """
        self.state['step_status'][step] = True
        self.logger.info(f"Step completed: {step}")
        
        # Trigger real-time streaming output
        if temp_output_dir and step in self.STEP_OUTPUT_DIRS:
            try:
                self.stream_step_output(step, temp_output_dir)
                
                # Record streamed steps
                if step not in self.state['output_streaming']['streamed_steps']:
                    self.state['output_streaming']['streamed_steps'].append(step)
                    
            except Exception as e:
                self.logger.error(f"Streaming output failed (step: {step}): {e}")
        
        self.save_state()
    
    def mark_subject_completed(self, step: str, subject_id: str, subject_output_path: Optional[Path] = None):
        """Mark subject step as completed
        
        Args:
            step: Step name
            subject_id: Subject ID
            subject_output_path: Subject output file path (for real-time streaming)
        """
        if step not in self.SUBJECT_LEVEL_STEPS:
            self.logger.warning(f"Step {step} does not support subject-level tracking")
            return
        
        completed_subjects = self.state['subject_progress'][step]
        if subject_id not in completed_subjects:
            completed_subjects.append(subject_id)
            self.logger.info(f"Subject completed (step: {step}, ID: {subject_id})")
            
            # Trigger subject-level streaming output
            if subject_output_path:
                try:
                    self.stream_subject_output(step, subject_id, subject_output_path)
                except Exception as e:
                    self.logger.error(f"Streaming subject output failed (step: {step}, ID: {subject_id}): {e}")
            
            # Periodically save state (every 5 subjects)
            if len(completed_subjects) % 5 == 0:
                self.save_state()
    
    def stream_step_output(self, step: str, temp_dir: Path):
        """Stream step output in real-time to final directory
        
        Args:
            step: Step name
            temp_dir: Temporary directory path
        """
        if step not in self.STEP_OUTPUT_DIRS:
            return
        
        target_dir = self.intermediate_dir / self.STEP_OUTPUT_DIRS[step]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if temp_dir.exists():
            self.logger.info(f"Streaming step output: {temp_dir} -> {target_dir}")
            
            # Copy all files
            for src_file in temp_dir.rglob('*'):
                if src_file.is_file():
                    rel_path = src_file.relative_to(temp_dir)
                    target_file = target_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, target_file)
        else:
            self.logger.warning(f"Temporary directory does not exist: {temp_dir}")
    
    def stream_subject_output(self, step: str, subject_id: str, subject_path: Path):
        """Stream subject output in real-time to final directory
        
        Args:
            step: Step name
            subject_id: Subject ID
            subject_path: Subject output file path
        """
        if step not in self.STEP_OUTPUT_DIRS:
            return
        
        target_dir = self.intermediate_dir / self.STEP_OUTPUT_DIRS[step]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if subject_path.exists():
            target_file = target_dir / subject_path.name
            self.logger.debug(f"Streaming subject output: {subject_path} -> {target_file}")
            shutil.copy2(subject_path, target_file)
        else:
            self.logger.warning(f"Subject output file does not exist: {subject_path}")
    
    def validate_step_output(self, step: str, output_path: Path) -> bool:
        """Validate step output integrity
        
        Args:
            step: Step name
            output_path: Output path
            
        Returns:
            Whether validation passes
        """
        if not output_path.exists():
            return False
        
        # Check if there are NII files
        nii_files = list(output_path.glob('*.nii.gz')) + list(output_path.glob('*.nii'))
        
        if step in self.SUBJECT_LEVEL_STEPS:
            # Subject-level steps need to check that each subject has output
            expected_count = len(self.state['input_info']['subject_ids'])
            return len(nii_files) >= expected_count
        else:
            # Other steps just need to have files
            return len(nii_files) > 0
    
    def verify_subject_output(self, step: str, subject_id: str, output_path: Path) -> bool:
        """Validate specific subject output integrity
        
        Args:
            step: Step name
            subject_id: Subject ID
            output_path: Output file path
            
        Returns:
            Whether validation passes
        """
        if not output_path.exists():
            return False
        
        # Check file size (basic integrity check)
        if output_path.stat().st_size == 0:
            return False
        
        # Can add more validation logic, such as:
        # - Check if NII file can be read correctly
        # - Check if image dimensions are reasonable
        # - Check label value ranges, etc.
        
        return True
    
    def handle_corruption(self, step: str, subject_id: Optional[str] = None):
        """Handle output corruption situations
        
        Args:
            step: Step name
            subject_id: Subject ID (optional)
        """
        if subject_id:
            # Remove subject's completion status
            completed_subjects = self.state['subject_progress'].get(step, [])
            if subject_id in completed_subjects:
                completed_subjects.remove(subject_id)
                self.logger.warning(f"Removed corrupted subject progress: {step} - {subject_id}")
        else:
            # Reset entire step
            self.state['step_status'][step] = False
            if step in self.SUBJECT_LEVEL_STEPS:
                self.state['subject_progress'][step] = []
            self.logger.warning(f"Reset corrupted step: {step}")
        
        self.save_state()
    
    def generate_progress_report(self) -> str:
        """Generate detailed progress report
        
        Returns:
            Formatted progress report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NPC Segmentation Pipeline Progress Report")
        report_lines.append("=" * 80)
        
        # Basic information
        report_lines.append(f"Pipeline ID: {self.state['pipeline_id']}")
        report_lines.append(f"Start Time: {self.state['start_time']}")
        report_lines.append(f"Last Updated: {self.state['last_updated']}")
        report_lines.append(f"Total Subjects: {self.state['input_info']['total_subjects']}")
        
        # Processing time
        start_time = datetime.fromisoformat(self.state['start_time'])
        last_updated = datetime.fromisoformat(self.state['last_updated'])
        processing_time = (last_updated - start_time).total_seconds()
        report_lines.append(f"Processing Time: {processing_time:.1f} seconds")
        
        report_lines.append("")
        
        # Step progress
        report_lines.append("Step Progress:")
        report_lines.append("-" * 40)
        
        completed_steps = sum(1 for status in self.state['step_status'].values() if status)
        total_steps = len(self.PIPELINE_STEPS)
        overall_progress = (completed_steps / total_steps) * 100
        
        report_lines.append(f"Overall Progress: {completed_steps}/{total_steps} ({overall_progress:.1f}%)")
        report_lines.append("")
        
        for step in self.PIPELINE_STEPS:
            status = "✓" if self.state['step_status'][step] else "✗"
            step_name = step.replace('_', ' ').title()
            report_lines.append(f"  {status} {step_name}")
            
            # Subject-level detailed information
            if step in self.SUBJECT_LEVEL_STEPS:
                completed_subjects = len(self.state['subject_progress'][step])
                total_subjects = self.state['input_info']['total_subjects']
                if total_subjects > 0:
                    subject_progress = (completed_subjects / total_subjects) * 100
                    report_lines.append(f"    Subject Progress: {completed_subjects}/{total_subjects} ({subject_progress:.1f}%)")
        
        report_lines.append("")
        
        # Streaming output status
        report_lines.append("Streaming Output Status:")
        report_lines.append("-" * 40)
        streamed_steps = self.state['output_streaming']['streamed_steps']
        if streamed_steps:
            for step in streamed_steps:
                step_name = step.replace('_', ' ').title()
                report_lines.append(f"  ✓ {step_name}")
        else:
            report_lines.append("  No streamed steps")
        
        report_lines.append("")
        
        # Recovery information
        resume_step, remaining_subjects = self.get_resume_point()
        if resume_step != 'output_completed':
            report_lines.append("Recovery Information:")
            report_lines.append("-" * 40)
            step_name = resume_step.replace('_', ' ').title()
            report_lines.append(f"Next Step: {step_name}")
            if remaining_subjects:
                report_lines.append(f"Remaining Subjects: {len(remaining_subjects)}")
                if len(remaining_subjects) <= 10:
                    report_lines.append(f"Subject IDs: {', '.join(remaining_subjects)}")
        else:
            report_lines.append("🎉 All steps completed!")
        
        report_lines.append("=" * 80)
        
        return '\n'.join(report_lines)
    
    def cleanup_progress(self):
        """Clean up progress files (for restart)"""
        if self.state_file_path.exists():
            self.state_file_path.unlink()
            self.logger.info("Cleaned up progress file")
        
        # Optional: clean up intermediate output directory
        if self.intermediate_dir.exists():
            shutil.rmtree(self.intermediate_dir)
            self.logger.info("Cleaned up intermediate output directory")
        
        # Reinitialize state
        self.state = self._create_new_state()
    
    def get_processing_time(self) -> float:
        """Get total processing time (seconds)
        
        Returns:
            Processing time (seconds)
        """
        start_time = datetime.fromisoformat(self.state['start_time'])
        last_updated = datetime.fromisoformat(self.state['last_updated'])
        return (last_updated - start_time).total_seconds()
    
    def export_progress_log(self, log_file: Path):
        """Export progress log to file
        
        Args:
            log_file: Log file path
        """
        report = self.generate_progress_report()
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Progress report exported to: {log_file}")