from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.solvers import BinaryClassificationSolver
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer
from typing import Union, Optional
from pathlib import Path
import torch.nn as nn

PathLike = Union[str, Path]

class rAIController(PMIController):
    r""""""
    def override_cfg(self, override_file: PathLike):
        r"""Additional overrides that is specific with rAI configurations"""
        super(rAIController, self).override_cfg(override_file)


        # If MaxVit is used size required is [320, 320, 20]
        if self.net_name.find('maxvit') >= 0:
            self.data_loader_cfg.sampler_kwargs = dict(
                patch_size = [320,320,20]
            )

            self.solver_cls = BinaryClassificationSolver
            self.inferencer_cls = BinaryClassificationInferencer
            self._data_loader_inf_cfg.augmentation = './maxvit_inf_transform.yaml'

        # * Save some memory by using less batch-size during validation
        if self.solver_cfg.rAI_fixed_mode >= 3:
            self.solver_cfg.batch_size_val = self.solver_cfg.batch_size // 4

        # * Pass attribute from controller to solver.
        # Substitute {fold_code} here because _pre_process_flags only covers rAI_pretrained_swran,
        # and override_cfg runs before exec() / _pre_process_flags.
        _pretrain_path = getattr(self.cfg, 'rAI_pretrained_CNN', None)
        if isinstance(_pretrain_path, str) and self.fold_code:
            _pretrain_path = _pretrain_path.replace('{fold_code}', str(self.fold_code))
        self.solver_cfg.rAI_pretrained_CNN = _pretrain_path
        self.solver_cfg.net_name = self.net_name


    def exec(self):
        r"""Because the network might be redefined by guild after :func:`override_cfg` is called, the mode is explicitly
        set here again since mode 0 using `BinaryClassificationSolver` doesn't have the callback to set it during run.
        """
        try:
            self.solver_cfg.net.set_mode(self.solver_cfg.rAI_fixed_mode)
        except AttributeError:
            self._logger.warning('Network mode not set.')
        super(rAIController, self).exec()
