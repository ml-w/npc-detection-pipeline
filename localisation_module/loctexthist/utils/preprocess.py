import mnts
from mnts.scripts.normalization import run_graph_train, run_graph_inference
from typing import Any, Union, Optional
from pathlib import Path
import SimpleITK as sitk


class NPCSegmentPreprocesser:
    def __init__(self, normalization_graph: Union[str, Path], state_dir: Union[str, Path],
                 training: Optional[bool] = False,
                 ):
        self.normalizaiton_graph = Path(normalization_graph)
        self.training = training
        self.state_dir = Path(state_dir)

        self._input_dir = None
        self._output_dir = None

        if not self.normalizaiton_graph.is_file():
            raise FileNotFoundError(f"Cannot find normalization graph under: {normalization_graph}")
        else:
            self.graph = mnts.filters.MNTSFilterGraph.CreateGraphFromYAML(self.normalizaiton_graph)

        # set normalization for training or testing
        if self.training:
            self.run_func = run_graph_train
        else:
            self.run_func = run_graph_inference

        # check state_dir
        if not self.state_dir.is_dir():
            raise AttributeError(f"State dir {self.state_dir} does not exist")
        else:
            if not self.training and len(list(self.state_dir.glob("*"))) == 0:
                raise AttributeError("State dir must contain states.")

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @input_dir.setter
    def input_dir(self, x: Union[str, Path]) -> None:
        x = Path(x)
        if not (x.is_dir() or x.is_file()):
            raise FileNotFoundError(f"Cannot found {str(x)}")
        self._input_dir = x

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, x) -> None:
        x = Path(x)
        if not x.is_dir():
            warnings.warn(f"Specified output directory {x} does not exist! Make sure to create it before proceed.")
        self._output_dir = x


    def normalize_image(self, im: Union[str, sitk.Image]) -> sitk.Image:
        """Load image and Normalize a single image input"""
        raise NotImplementedError

    def exec(self) -> None:
        """Use console entry of mnts to run path normalization"""
        normalized_dir = self.output_dir
        if not normalized_dir.is_dir():
            normalized_dir.mkdir(parents=True)

        # check if there's really any nii files
        nii_files = list(self.input_dir.rglob("*nii*"))
        nii_files.sort()
        if len(nii_files) == 0:
            raise FileNotFoundError(f"Nothing is found in the temporary directory.")

        # run noramlization
        run_graph_inference(f"-i {str(self.input_dir)} -o {str(self.output_dir)} "
                            f"-f {str(self.normalizaiton_graph)} "
                            f"--state-dir {self.state_dir}".split())

