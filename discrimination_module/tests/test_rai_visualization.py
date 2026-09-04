import unittest
import tempfile
import torchio
import SimpleITK as sitk
import numpy as np
import json
from rAIdiologist.utils.visualization_rAIdiologist import *

class Test_visualization_rAIdiologist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._logger = MNTSLogger('.', logger_name=cls.__name__, verbose=True, keep_file=False, log_level='debug')
        cls.tmp_img_dir = tempfile.TemporaryDirectory()
        cls.tmp_img_path = Path(cls.tmp_img_dir.name)

        out_json = {}
        # create dummy data
        for i in range(3):
            num_slices = 30
            save_path = cls.tmp_img_path.joinpath(f'IMG{i}.nii.gz')
            tmp_im = sitk.GetImageFromArray(np.zeros([512, 512, num_slices]))
            sitk.WriteImage(tmp_im, str(save_path))

            vec = np.zeros([num_slices, 5])
            vec[..., -2] = np.arange(num_slices) # index
            vec[..., -1] = 0                     # direction
            vec[..., 1] = np.sin(vec[..., -2])   # lstm prediction
            vec[..., 0] = np.cos(vec[..., -2])   # cnn prediction
            vec[..., 2] = np.cos(vec[..., -2])   # confidence
            out_json[f'IMG{i}'] = vec.tolist()

        cls.json_file_path = cls.tmp_img_path.joinpath("prediction.json")
        json.dump(out_json, cls.json_file_path.open('w'))

    @classmethod
    def tearDownClass(cls) -> None:
        MNTSLogger.cleanup()
        cls.tmp_img_dir.cleanup()

    def setUp(self):
        super(Test_visualization_rAIdiologist, self).setUp()
        self.img_dir = self.tmp_img_path
        self.json = json.load(Path(self.__class__.json_file_path).open('r'))
        self.image = tio.ScalarImage(str(self.img_dir.joinpath('IMG0.nii.gz')))[tio.DATA].squeeze()
        self.cnnprediction, self.prediction, self.conf, self.indices, self.direction = unpack_json(self.json, 'IMG0')
        self.temp_out_dir  = tempfile.TemporaryDirectory()
        self.temp_out_path = Path(self.temp_out_dir.name)

    def test_mark_slice(self):
        x1 = make_marked_slice(
            self.image[..., 15],
            self.cnnprediction,
            self.prediction,
            self.conf,
            self.indices,
            self.direction
        )
        x2 = make_marked_slice(
            self.image[..., 15],
            self.cnnprediction,
            self.prediction,
            self.conf,
            self.indices,
            self.direction,
            vert_line=15
        )
        x3 = make_marked_slice(
            self.image[..., 15],
            self.cnnprediction,
            self.prediction,
            self.conf,
            self.indices,
            self.direction,
            imshow_kwargs={'cmap':'jet'}
        )

        self.assertTrue(x1.shape == x2.shape == x3.shape)

    def test_mark_stack(self):
        s = mark_image_stacks(self.image,
                              self.cnnprediction,
                              self.prediction,
                              self.conf,
                              self.indices,
                              self.direction)
        marked_stack_2_grid(s, self.temp_out_path / 'test.png')

    def test_mark_stack_wo_cnnprediction(self):
        s = mark_image_stacks(self.image,
                              None,
                              self.prediction,
                              self.conf,
                              self.indices,
                              self.direction)
        marked_stack_2_grid(s, self.temp_out_path / 'test.png')

    @unittest.skip("Cannot be tested with other tests")
    def test_label_images_in_dir(self):
        temp_dir = tempfile.TemporaryDirectory()
        label_images_in_dir(self.img_dir, self.json, temp_dir.name, idGlobber="MRI_[0-9]+")
        print(list(Path(temp_dir.name).iterdir()))
        self.assertEqual(3, len(list(Path(temp_dir.name).iterdir())))
        temp_dir.cleanup()