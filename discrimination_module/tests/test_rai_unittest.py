import sys
import os
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

from rAIdiologist.config.network.rAIdiologist import *
from rAIdiologist.config.network.slicewise_ran import *
from rAIdiologist.config.network.old.old_swran import SlicewiseAttentionRAN_old
from rAIdiologist.config.network.ViT3d.models import ViTVNetImg2Pred, CONFIGS
from rAIdiologist.config.network.cnn import *
from rAIdiologist.config.rAIdiologistCFG import *
from rAIdiologist.solvers import *
import unittest
import torch
from einops.layers.torch import Rearrange
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.controller import PMIController
from pytorch_model_summary.model_summary import summary

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        if self.__class__.__name__ == 'Test3DNetworks':
            self.skipTest('Base test class')
        num_slice = 30
        num_data = 4
        torch.random.manual_seed(30)
        self.sample_input_big = torch.rand(num_data, 1, 512, 512, num_slice).cuda()
        self.sample_input = torch.rand(num_data, 1, 128, 128, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 128, 128, num_slice).cuda()
        self.sample_input_3d = torch.rand(num_data, 1, 128, 128, 128).cuda()
        self.sample_input_3d_size1 = torch.rand(1, 1, 128, 128, 128).cuda()
        self.sample_input[0, ..., 14::].fill_(0)
        self.sample_seg = torch.zeros_like(self.sample_input).cuda()
        self.sample_seg[0, 0, 50, 50, 10:20].fill_(1)
        self.sample_seg[1, 0, 50, 50, 8:15].fill_(1)
        self.sample_seg_size1 = torch.zeros_like(self.sample_input_size1).cuda()
        self.sample_seg_size1[0, 0, 50, 50, 10:20].fill_(1)
        self.expect_nonzero = torch.zeros([num_data, 1, num_slice], dtype=bool)
        self.expect_nonzero[0, ..., 9:21] = True
        self.expect_nonzero[1, ..., 7:16] = True
        self.EXPECTED_DIM = 2

    def expect_dim(self, out, expected_dim):
        msg = f"Dim test failed, expected dim = {expected_dim}, got shape: {out.shape}"
        self.assertEqual(expected_dim, out.dim(), msg)

    def test_input(self):
        with torch.no_grad():
            out = self.net(self.sample_input)
            try:
                # Try to print the summary for convenience
                summary(self.net, self.sample_input, show_input=True, print_summary=True, max_depth=2)
            except:
                pass

            # Expected output: torch.Tensor or Tuple[torch.Tensor]
            if isinstance(out, torch.Tensor):
                self.expect_dim(out, self.EXPECTED_DIM)
            elif isinstance(out, (list, tuple)):
                pred, conf = out
                self.expect_dim(pred, self.EXPECTED_DIM)
            else:
                raise TypeError("Expected torch tensor output")

    def test_input_bsize_1(self):
        with torch.no_grad():
            out = self.net(self.sample_input_size1)
            # Expected output: torch.Tensor or Tuple[torch.Tensor]
            if isinstance(out, torch.Tensor):
                self.expect_dim(out, self.EXPECTED_DIM)
            elif isinstance(out, (list, tuple)):
                pred, conf = out
                self.expect_dim(pred, self.EXPECTED_DIM)
            else:
                raise TypeError("Expected torch tensor output")


class TestOldSWRAN(Test3DNetworks):
    def setUp(self) -> None:
        super(TestOldSWRAN, self).setUp()
        self.net = SlicewiseAttentionRAN_old(1, 1).cuda()


class TestRAIdiologist(Test3DNetworks):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = rAIdiologist(1, record=False).cuda()

    def test_rAIdiologist_modes(self):
        with torch.no_grad():
            for i in range(6):
                try:
                    self.net.set_mode(i)
                    self.assertTrue(self.net._mode == i)
                    self.test_input()
                    self.test_input_bsize_1()
                    print(f"Mode {i} passed.")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")

    def test_rAIdiologist_recordon(self):
        self.net.RECORD_ON = True
        self.net.set_mode(5)
        self.net = self.net.eval()
        with torch.no_grad():
            try:
                out = self.net(self.sample_input)
                self.assertNotEqual(0, len(self.net.get_playback()), "Playback length is zero")
                out = self.net(self.sample_input_size1)
            except Exception as e:
                self.fail(f"Original message: {e}")
            self.assertNotEqual(0, len(self.net.get_playback()), "Play back is empty!")


class TestRAN25D(Test3DNetworks):
    def setUp(self) -> None:
        super(TestRAN25D, self).setUp()
        self.net = RAN_25D(1, 1).cuda()


class TestSWRAN(Test3DNetworks):
    def setUp(self) -> None:
        super(TestSWRAN, self).setUp()
        self.net = SlicewiseAttentionRAN(1, 1).cuda()


# ==== TEST RAIs ====
@unittest.skip("v3 is broken")
class TestRAIv3(TestRAIdiologist):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = create_rAIdiologist_v3().cuda()
        del self.EXPECTED_DIM


class TestRAIv5(TestRAIdiologist):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = create_rAIdiologist_v5().cuda()

    def expect_dim(self, out, expected_dim):
        if self.net._mode != 0:
            self.EXPECTED_DIM = 3
        super().expect_dim(out, self.EXPECTED_DIM)


class TestRAIv5_1(TestRAIv5):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = create_rAIdiologist_v5_1().cuda()


class TestRAIv5_1_focused(TestRAIv5):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = create_rAIdiologist_v5_1_focused().cuda()


class TestViT(Test3DNetworks):
    def setUp(self) -> None:
        super(TestViT, self).setUp()
        config = CONFIGS['ViT3d-Img2Pred']
        # After encode, patch-size is (20, 20, 25), `config.patches.grid` must be able to fully divide the patch
        config.patches.grid = (10, 10, 25)
        self.sample_input = torch.rand(4, 1, 320, 320, 25).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 320, 320, 25).cuda()
        self.net = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25)).cuda()


class TestRAIController(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._logger = MNTSLogger('.', logger_name='unittest', log_level='debug', keep_file=False, verbose=True)
        MNTSLogger.set_global_log_level('debug')

    def setUp(self) -> None:
        super(TestRAIController, self).setUp()
        self.controller.debug_mode = True
        self.controller.solver_cfg.num_of_epochs = 2
        self.controller.solver_cfg.batch_size = 1

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        cfg = MyControllerCFG()
        cfg.data_loader_val_cfg.augmentation = '../rAIdiologist/config/rAIdiologist_transform_inf.yaml'
        cfg.plotting = False # Turn off plotting for controller test
        self.controller = PMIController(cfg)
        pass

    def test_s1_create(self):
        pass

    def test_s2_fit(self):
        self.controller.exec()

    def test_s3_inference(self):
        self.controller.run_mode = False
        self.controller.exec()

    def test_s4_validation(self):
        with torch.no_grad():
            self.controller.debug_validation = True
            self.controller.solver_cfg._last_epoch_loss = 1E32
            self.controller.exec()


class TestRAIFocusedController(TestRAIController):
    def setUp(self) -> None:
        super(TestRAIController, self).setUp()
        self.controller.debug_mode = True
        self.controller.solver_cfg.num_of_epochs = 2
        self.controller.solver_cfg.batch_size = 1

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        cfg = rAIControllerFocusedCFG()
        cfg.data_loader_val_cfg.augmentation = '../rAIdiologist/config/rAIdiologist_transform_focused_inf.yaml'
        cfg.plotting = False
        self.controller = PMIController(cfg)
        pass
    
    
class TestOtherCNNs(Test3DNetworks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nets = [
            get_ResNet3d(101),
            get_vgg16()
        ]
        self._logger = MNTSLogger['TestOtherCNNs']
        
    def test_input(self):
        for n in self.nets:
            self._logger.info(f"Testing network: {type(n)}")
            self.net = n.cuda()
            super().test_input()
            del self.net
            
    def test_input_bsize_1(self):
        for n in self.nets:
            self._logger.info(f"Testing network: {type(n)}")
            self.net = n.cuda()
            super().test_input_bsize_1()
            del self.net