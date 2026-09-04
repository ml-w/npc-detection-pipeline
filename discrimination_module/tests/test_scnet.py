import tempfile

from pytorch_med_imaging.controller import PMIController
from rAIdiologist.config.network.scnet import SCDenseNet
from rAIdiologist.config.loss.scdensenet_loss import DualLoss
from rAIdiologist.config.SCDNetCFG import *
from pytorch_med_imaging.pmi_data import DataLabel
from mnts.mnts_logger import MNTSLogger
import copy
import unittest
import torch
import torchio as tio
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_model_summary

class Test_SCNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a dummy checkpoint
        cls.temp_dummy_checkpoint = tempfile.NamedTemporaryFile(suffix='.pt')
        net = SCDenseNet()
        torch.save(net.state_dict(), cls.temp_dummy_checkpoint)
        cls._logger = MNTSLogger(".", 'pytest', verbose=True, keep_file=False, log_level='debug')
        cls._logger.set_log_level('debug')
        MNTSLogger.set_global_log_level('debug')

    @classmethod
    def tearDownClass(cls):
        cls.temp_dummy_checkpoint.close()

    def setUp(self):
        num_slice = 16
        num_data = 4
        torch.random.manual_seed(30)
        self.sample_input_big = torch.rand(num_data, 1, 384, 384, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 384, 384, num_slice).cuda()

        self.input_dir = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized'
        self.gt_seg_dir = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2_SCDensnet/T2WFS_TRA'
        self.gt_data = DataLabel.from_csv('./NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv')
        self.gt_data.set_target_column('is_malignant')

        self.data_loader_cfg = copy.deepcopy(data_loader)
        self.data_loader_cfg.augmentation = './../rAIdiologist/config/SCDNet_transform_train.yaml'
        self.data_loader_cfg.debug_mode = True
        self.data_loader_cfg.input_data['input'] = self.input_dir
        self.data_loader_cfg.input_data['gt_seg'] = self.gt_seg_dir
        self.data_loader_cfg.input_data['gt'] = self.gt_data
        self.data_loader_cfg.sampler = 'weighted' # Unset sampler to load the whole image
        self.data_loader_cfg.sampler_kwargs = dict(
            patch_size=[384, 384, 16]
        )

        self.data_loader_inf_cfg = copy.deepcopy(data_loader_test)
        self.data_loader_inf_cfg.augmentation = './../rAIdiologist/config/SCDNet_transform_inf.yaml'
        self.data_loader_inf_cfg.debug_mode = True
        self.data_loader_inf_cfg.input_data['input'] = self.input_dir
        self.data_loader_inf_cfg.input_data['gt_seg'] = self.gt_seg_dir
        self.data_loader_inf_cfg.input_data['gt'] = self.gt_data

        self.solver_cfg = SCDenseNetSolverCFG()
        self.solver_cfg.debug_mode = True
        self.solver_cfg.batch_size = 2


    def test_scnet(self):
        with torch.no_grad():
            scnet = SCDenseNet().cuda()
            x_cls, x_seg = scnet(self.sample_input_big)

            # expects two outputs, one segmentation map and one benign/cancer prediction
            self.assertTupleEqual(x_seg.shape, self.sample_input_big.shape)
            self.assertTupleEqual(x_cls.shape, (x_seg.shape[0], 1))

            # try case with one sample
            x_cls, x_seg = scnet(self.sample_input_size1)
            self.assertTupleEqual(x_seg.shape, self.sample_input_size1.shape)
            self.assertTupleEqual(x_cls.shape, (1, 1))

    def test_scnet_dual_loss(self):
        scnet = SCDenseNet().cuda()
        scnet_dual_loss = DualLoss()

        mock_input_pred = (
            torch.rand([self.sample_input_big.shape[0], 1]),
            torch.rand(self.sample_input_big.shape)
        )
        mock_input_gt = (
            torch.randint(0, 1, [self.sample_input_big.shape[0], 1]).float(),
            torch.randint(0, 1, self.sample_input_big.shape)
        )
        scnet_dual_loss(*mock_input_pred, *mock_input_gt)

    def test_solver_step(self):
        loader = PMITorchioDataLoader(self.data_loader_cfg)
        solver = SCDenseNetSolver(self.solver_cfg)
        self.solver_cfg.batch_size = 2
        solver.set_data_loader(loader)
        for mb in solver.data_loader:
            solver.current_uid = mb.get('uid', None)
            s, g = solver._unpack_minibatch(mb, solver.unpack_key_forward)
            out, loss = solver.step(s, g)
            self.assertIsInstance(out, tuple)
            break

    def test_solver_fit(self):
        loader = PMITorchioDataLoader(self.data_loader_cfg)
        loader_inf = PMITorchioDataLoader(self.data_loader_inf_cfg)
        solver = SCDenseNetSolver(self.solver_cfg)
        solver.set_data_loader(loader, loader_inf)
        solver.batch_size = 2
        solver.max_step = 3
        solver.solve_epoch(0)

    def test_controller_fit(self):
        cfg = SCDControllerCFG()
        cfg.plotting = False
        cfg.debug_mode = True
        cfg.solver_cfg.batch_size = 2
        controller = PMIController(cfg)
        controller.exec()

    def test_inferencer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            loader_inf = PMITorchioDataLoader(self.data_loader_inf_cfg)
            self.solver_cfg.cp_load_dir = self.temp_dummy_checkpoint.name
            solver = SCDenseNetInferencer(self.solver_cfg)
            solver.set_data_loader(loader_inf)
            solver.output_dir = Path(tempdir)
            solver.batch_size = 2
            solver.max_step = 3
            solver._write_out()
            solver.display_summary()
            self._logger.info(f"{len(list(Path(tempdir).rglob('*')))}")

    def test_dataloader_train(self):
        loader = PMITorchioDataLoader(self.data_loader_cfg)
        queue = loader._load_data_set_training()

        self._logger.debug(f"{queue = }")
        self._logger.debug(f"{queue.shuffle_subjects = }")
        self._logger.debug(f"{queue.shuffle_patches = }")

        for i, mb in enumerate(queue):
            self.assertIn('input', mb)
            self.assertIn('gt', mb)
            self.assertIn('gt_seg', mb)
            self.assertIn('uid', mb)
            self._logger.debug(f"{mb['input'].shape = }, {mb['gt']}, {mb['gt_seg'].shape}, {mb['uid'] = }")
            if i > 10:
                break

        torchLoader = DataLoader(queue,
                                batch_size=2,
                                shuffle=queue.shuffle_subjects,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=False)
        for i, mb in enumerate(torchLoader):
            self.assertIn('input', mb)
            self.assertIn('gt', mb)
            self.assertIn('gt_seg', mb)
            self.assertIn('uid', mb)
            self._logger.debug(f"\n{mb['input'][tio.DATA].shape = }, \n{mb['gt'].flatten()}, \b{mb['gt_seg'][tio.DATA].shape}, \n{mb['uid'] = }")
            if i > 10:
                break

    def test_dataloader_inference(self):
        loader = PMITorchioDataLoader(self.data_loader_cfg)
        queue = loader._load_data_set_inference()

        self._logger.debug(f"{queue = }")
        self._logger.debug(f"{queue.shuffle_subjects = }")
        self._logger.debug(f"{queue.shuffle_patches = }")

        for i, mb in enumerate(queue):
            self.assertIn('input', mb)
            self.assertIn('gt', mb)
            self.assertIn('gt_seg', mb)
            self.assertIn('uid', mb)
            self._logger.debug(f"{mb['input'].shape = }, {mb['gt']}, {mb['gt_seg'].shape}, {mb['uid'] = }")
            if i > 10:
                break

        torchLoader = DataLoader(queue,
                                batch_size=2,
                                shuffle=queue.shuffle_subjects,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=False)
        for i, mb in enumerate(torchLoader):
            self.assertIn('input', mb)
            self.assertIn('gt', mb)
            self.assertIn('gt_seg', mb)
            self.assertIn('uid', mb)
            self._logger.debug(f"\n{mb['input'][tio.DATA].shape = }, \n{mb['gt'].flatten()}, \b{mb['gt_seg'][tio.DATA].shape}, \n{mb['uid'] = }")
            if i > 10:
                break