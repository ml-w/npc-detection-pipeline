import unittest
import torch
from rAIdiologist.config.network.cnn import get_ResNet3d, get_vgg16, get_vgg
import torch.nn as nn
from rAIdiologist.config.network.densenet3d import get_densenet3d_121, get_densenet3d
from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v1, create_rAIdiologist_v2, create_rAIdiologist_v3, create_rAIdiologist_v4, create_rAIdiologist_v41, create_rAIdiologist_v42, create_rAIdiologist_v43, create_rAIdiologist_v5, create_rAIdiologist_v5_1, create_rAIdiologist_v5_1_focused, create_old_rAI, create_old_rAI_rmean
from rAIdiologist.config.network.scnet import SCDenseNet
from rAIdiologist.config.network.slicewise_ran import RAN_25D, SlicewiseAttentionRAN, SWRAN_Block, SWRAN_Block_Focused
from rAIdiologist.config.network.vit import ViT
# Import other models as needed from rAIdiologist.config.network/

class TestNetworkForward(unittest.TestCase):
    """
    Base class for testing network forward passes.
    """
    def setUp(self):
        self.model = None # To be set by child classes
        self.input_shape = (1, 1, 320, 320, 25) # Default input shape (batch, channels, depth, height, width)
        self.output_dim = 1 # Default output dimension for binary classification

    def _test_forward_pass(self, model, input_shape, expected_output_dim):
        """
        Helper method to test the forward pass of a given model.
        """
        with torch.no_grad():
            self.assertIsNotNone(model, "Model should be initialized.")

            # Test with batch size 1
            input_tensor_bs1 = torch.randn(input_shape)
            output_bs1 = model(input_tensor_bs1)
            self.assertEqual(output_bs1.shape[-1], expected_output_dim,
                             f"Batch size 1: Expected output dimension {expected_output_dim}, got {output_bs1.shape[-1]}")

            # Test with batch size 2
            input_tensor_bs2 = torch.randn((2,) + input_shape[1:])
            output_bs2 = model(input_tensor_bs2)
            self.assertEqual(output_bs2.shape[-1], expected_output_dim,
                             f"Batch size 2: Expected output dimension {expected_output_dim}, got {output_bs2.shape[-1]}")

    @unittest.skip("Skip test_forward method in base class")
    def test_forward(self):
        """
        Placeholder for actual forward pass test in child classes.
        """
        raise NotImplementedError("Subclasses must implement test_forward method.")

class TestResNet3d101(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_ResNet3d(101)

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestVGG16(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_vgg16()

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestVGG11(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_vgg(size='11')

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestVGG13(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_vgg(size='13')

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestVGG19(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_vgg(size='19')

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestDenseNet3D121(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = get_densenet3d_121()

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)


class TestOldRAI(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = create_old_rAI()

    def test_forward(self):
        self.model.set_mode(5)
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestOldRAIRMean(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = create_old_rAI_rmean()

    def test_forward(self):
        self.model.set_mode(5)
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestSCDenseNet(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = SCDenseNet()
        # SCDenseNet returns two outputs: classification and segmentation
        # We are only testing the classification output here.

    def _test_forward_pass(self, model, input_shape, expected_output_dim):
        self.assertIsNotNone(model, "Model should be initialized.")
        
        # Test with batch size 1
        input_tensor_bs1 = torch.randn(input_shape)
        output_bs1, _ = model(input_tensor_bs1) # SCDenseNet returns two outputs
        self.assertEqual(output_bs1.shape[-1], expected_output_dim,
                         f"Batch size 1: Expected output dimension {expected_output_dim}, got {output_bs1.shape[-1]}")
        
        # Test with batch size 2
        input_tensor_bs2 = torch.randn((2,) + input_shape[1:])
        output_bs2, _ = model(input_tensor_bs2) # SCDenseNet returns two outputs
        self.assertEqual(output_bs2.shape[-1], expected_output_dim,
                         f"Batch size 2: Expected output dimension {expected_output_dim}, got {output_bs2.shape[-1]}")

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestRAN25D(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = RAN_25D(in_ch=1, out_ch=1)

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestSlicewiseAttentionRAN(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = SlicewiseAttentionRAN(in_ch=1, out_ch=1)

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

class TestDenseNet3D(unittest.TestCase):
    def setUp(self):
        # Default input shape: (batch_size, channels, depth, height, width)
        self.input_shape = (1, 1, 320, 320, 320)
        self.num_classes = 10

    def _test_dense_net_3d_forward(self, model_depth, expected_output_channels):
        """Tests the forward pass of a DenseNet3D model."""
        model = get_densenet3d(model_depth=model_depth, num_classes=self.num_classes, in_channels=self.input_shape[1])
        model.eval()  # Set model to evaluation mode

        # Test with batch size 1
        input_tensor = torch.randn(self.input_shape)
        with torch.no_grad():
            output = model(input_tensor)

        self.assertEqual(output.shape[-1], expected_output_channels,
                         f"Failed for DenseNet3D-{model_depth} with batch size 1. Expected output channels: {expected_output_channels}, got: {output.shape[-1]}")

        # Test with batch size 4
        input_tensor_bs4 = torch.randn((4,) + self.input_shape[1:])
        with torch.no_grad():
            output_bs4 = model(input_tensor_bs4)

        self.assertEqual(output_bs4.shape[-1], expected_output_channels,
                         f"Failed for DenseNet3D-{model_depth} with batch size 4. Expected output channels: {expected_output_channels}, got: {output_bs4.shape[-1]}")

    def test_densenet3d_121(self):
        """Tests DenseNet3D-121 forward pass."""
        self._test_dense_net_3d_forward('121', self.num_classes)

    def test_densenet3d_169(self):
        """Tests DenseNet3D-169 forward pass."""
        self._test_dense_net_3d_forward('169', self.num_classes)

    def test_densenet3d_201(self):
        """Tests DenseNet3D-201 forward pass."""
        self._test_dense_net_3d_forward('201', self.num_classes)

    def test_densenet3d_264(self):
        """Tests DenseNet3D-264 forward pass."""
        self._test_dense_net_3d_forward('264', self.num_classes)

    def test_unsupported_depth(self):
        """Tests that an unsupported depth raises a ValueError."""
        with self.assertRaises(ValueError):
            get_densenet3d(model_depth='101')  # Assuming '101' is not a supported depth


class TestViT(TestNetworkForward):
    def setUp(self):
        super().setUp()
        self.model = ViT(channels=1, num_classes=1, image_size=320, patch_size=16,
                         dim=512, depth=6, heads=8, mlp_dim=1024, num_slices=25)
        self.model.eval()

    def test_forward(self):
        self._test_forward_pass(self.model, self.input_shape, self.output_dim)

    def test_forward_2d(self):
        model_2d = ViT(channels=1, num_classes=1, image_size=320, patch_size=16,
                       dim=512, depth=6, heads=8, mlp_dim=1024, num_slices=1)
        model_2d.eval()
        self._test_forward_pass(model_2d, (1, 1, 320, 320), self.output_dim)


if __name__ == '__main__':
    unittest.main()