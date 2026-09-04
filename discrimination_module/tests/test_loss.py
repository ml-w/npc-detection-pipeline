import unittest
import torch
from rAIdiologist.config.loss import *

class TestLossFunction(unittest.TestCase):
    def test_confidence_ce_loss(self):
        dummy_input = torch.rand([10, 128])
        dummy_module = torch.nn.Linear(128, 4)
        dummy_target = torch.randint(0, 1, [10])
        loss_func = ConfidenceCELoss(weight=torch.Tensor([0.5, 1.2]))

        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            dummy_module = dummy_module.cuda()
            dummy_target = dummy_target.cuda()
            loss_func = loss_func.cuda()

        o = dummy_module.forward(dummy_input)
        pred = o[:, :2]
        conf = torch.clip(torch.sigmoid(o[:, 2:]), 0.2, 0.8)

        out = loss_func((pred, conf), dummy_target)
        out.mean().backward()

        self.assertEqual(1, out.dim())

    def test_confidence_bce_loss(self):
        expected_input_ch = 3

        dummy_input = torch.rand([10, 128])
        dummy_module = torch.nn.Linear(128, expected_input_ch)
        dummy_target = torch.as_tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        loss_func = ConfidenceBCELoss(conf_weight=0.5, over_conf_weight=0.1, gamma=0.01, pos_weight=torch.Tensor([1.1]))

        # Test forward
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            dummy_module = dummy_module.cuda()
            dummy_target = dummy_target.cuda()
            loss_func = loss_func.cuda()

        o = dummy_module.forward(dummy_input)
        pred = o[:, :expected_input_ch]

        out = loss_func(pred, dummy_target.float())
        out.backward()
        self.assertEqual(0, out.dim())

        # Test logic
        correct_prediction = torch.as_tensor([-10, -10, -10, -10, 10, 10, 10, 10, 10, 10]).float().cuda()
        with torch.no_grad():
            l = loss_func(correct_prediction.reshape(-1, 1), dummy_target.float().reshape(-1, 1))
            self.assertAlmostEqual(l.cpu().item(), 0, delta=1E-4)

        # Test forward for 3 column input
        dummy_input = torch.as_tensor([
            [-10, 0.2, -10],
            [-10, 0.2, -10],
            [-10, 0.8, -10],
            [-10, 0.8, -10],
            [10, 0.8, 10],
            [10, 0.8, 10],
            [10, 0.8, 10],
            [10, 0.8, 10],
            [10, 0.8, 10],
            [10, 0.8, 10],
        ]).cuda()
        with torch.no_grad():
            l = loss_func(dummy_input.float(), dummy_target.float())
            print(l)



if __name__ == '__main__':
    unittest.main()
