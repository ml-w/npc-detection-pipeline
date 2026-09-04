from pytorch_med_imaging.pmi_data_loader.augmenter_factory import create_transform_compose
from torchio import Compose
import unittest
from pathlib import Path

class Test_CreateTransform(unittest.TestCase):
    def setUp(self):
        self.train_transform_file = Path("../rAIdiologist/config/rAIdiologist_transform_train.yaml")
        self.train_focused_transform_file = Path("../rAIdiologist/config/rAIdiologist_transform_focused_train.yaml")
        self.inf_transform_file = Path("../rAIdiologist/config/rAIdiologist_transform_inf.yaml")
        self.inf_focused_transform_file = Path("../rAIdiologist/config/rAIdiologist_transform_focused_inf.yaml")

    def test_train_transform(self):
        trans = create_transform_compose(self.inf_transform_file)
        print(trans)
        self.assertIsInstance(trans, Compose)

    def test_train_focused_transform(self):
        trans = create_transform_compose(self.train_focused_transform_file)
        print(trans)
        self.assertIsInstance(trans, Compose)

    def test_inf_transform(self):
        trans = create_transform_compose(self.inf_transform_file)
        print(trans)
        self.assertIsInstance(trans, Compose)

    def test_inf_focused_transform(self):
        trans = create_transform_compose(self.inf_focused_transform_file)
        print(trans)
        self.assertIsInstance(trans, Compose)