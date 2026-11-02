import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v5_1, rAIdiologist_Transformer
from rAIdiologist.config.network.lstm_rater import TransformerEncoderLayerWithAttn


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_net(flag: bool):
    torch.manual_seed(0)
    return create_rAIdiologist_v5_1(rai_dual_crop_inference=flag).to(_device())


def _rand(*shape):
    return torch.rand(*shape, device=_device())


def _capture_rnn_input(net, x):
    """Run ``net(x)`` and capture the (tokens, topslices) the transformer receives."""
    seen = {}

    def hook(_module, args):
        seen['tokens'] = tuple(args[0].shape)
        seen['top'] = args[1].detach().clone()

    handle = net.rnn.register_forward_pre_hook(hook)
    try:
        out = net(x)
    finally:
        handle.remove()
    return out, seen


def _master_pkg_root():
    """Package root of untouched master ``rAIdiologist`` (contains the inner package dir)."""
    candidates = [
        os.environ.get('RAIDIOLOGIST_MASTER_ROOT', ''),
        '/Users/lwong/Source/deep-learning-projects/rAIdiologist',
        os.path.expanduser('~/Source/deep-learning-projects/rAIdiologist'),
        '/home/lwong/Source/deep-learning-projects/rAIdiologist',
    ]
    for p in candidates:
        if p and os.path.isfile(os.path.join(p, 'rAIdiologist', 'config', 'network', 'rAIdiologist.py')):
            return p
    return None


_MASTER_FWD_SCRIPT = r'''
import sys
import torch
sys.path.insert(0, sys.argv[1])
from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v5_1
x = torch.load(sys.argv[2], map_location="cpu")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(dev)
torch.manual_seed(int(sys.argv[3]))
net = create_rAIdiologist_v5_1()
net.set_mode(int(sys.argv[4]))
if sys.argv[5] == "train":
    net.train()
else:
    net.eval()
net = net.to(dev)
torch.manual_seed(int(sys.argv[6]))
if sys.argv[5] == "train":
    y = net(x)
else:
    with torch.no_grad():
        y = net(x)
torch.save(y.detach().cpu(), sys.argv[7])
print("master_out.device=%s" % y.device)
'''


def _master_forward(x, seed_create, seed_fwd, mode, training):
    root = _master_pkg_root()
    if root is None:
        return None, 'master rAIdiologist.py not found (set RAIDIOLOGIST_MASTER_ROOT)'
    env = os.environ.copy()
    env['PYTHONPATH'] = root
    with tempfile.TemporaryDirectory() as td:
        xin = os.path.join(td, 'x.pt')
        yout = os.path.join(td, 'y.pt')
        torch.save(x.detach().cpu(), xin)
        proc = subprocess.run(
            [sys.executable, '-c', _MASTER_FWD_SCRIPT, root, xin,
             str(seed_create), str(mode), 'train' if training else 'eval',
             str(seed_fwd), yout],
            cwd=root, env=env, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            return None, 'master subprocess failed: %s%s' % (proc.stdout, proc.stderr)
        y = torch.load(yout, map_location='cpu')
        return y, proc.stdout.strip()


class TestDualCropInference(unittest.TestCase):
    def _assert_runtime_device(self, t):
        print('out.device=%s' % t.device, flush=True)
        self.assertEqual(t.device.type, _device().type)
        if torch.cuda.is_available():
            self.assertTrue(t.is_cuda, 'cuda is available but tensor is on %s' % t.device)

    # Point 1: flag default, device-agnostic SA, 17x17-style canvas, half-patch shift
    def test_flag_default_off(self):
        self.assertFalse(create_rAIdiologist_v5_1().rai_dual_crop_inference)

    def test_sa_from_playback_has_no_hardcoded_cuda(self):
        src = inspect.getsource(TransformerEncoderLayerWithAttn.sa_from_playback)
        self.assertNotIn('.cuda()', src)

    def test_two_view_canvas_no_overflow_at_h7(self):
        h, w, s = 7, 7, 4
        tokens = torch.rand(1, 1, s * 2 * h * w, device=_device())
        canvas = TransformerEncoderLayerWithAttn._two_view_to_canvas(tokens, h, w, s)
        self.assertEqual(tuple(canvas.shape), (1, 1, 2 * h + 1, 2 * w + 1, s))
        self._assert_runtime_device(canvas)

    def test_inplane_shift_zero_fill_no_wrap(self):
        x = torch.arange(1 * 1 * 4 * 4 * 3, dtype=torch.float, device=_device()).reshape(1, 1, 4, 4, 3) + 1
        y = rAIdiologist_Transformer._shift_inplane(x, 1, 2)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.all(y[:, :, :1] == 0))
        self.assertTrue(torch.all(y[:, :, :, :2] == 0))
        self.assertTrue(torch.equal(y[:, :, 1:, 2:], x[:, :, :-1, :-2]))
        self._assert_runtime_device(y)

    # Point 2: training scope guard (must equal master) and inference token/mask formula
    def test_training_flag_on_equals_flag_off(self):
        x = _rand(1, 1, 32, 32, 5)
        off = _make_net(False)
        off.set_mode(3)
        off.train()
        on = _make_net(True)
        on.set_mode(3)
        on.train()
        torch.manual_seed(1)
        out_off = off(x)
        torch.manual_seed(1)
        out_on = on(x)
        self.assertEqual(tuple(out_on.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(out_off, out_on))
        self._assert_runtime_device(out_on)
        self._assert_runtime_device(out_off)

    def test_training_flag_off_is_deterministic(self):
        x = _rand(1, 1, 32, 32, 5)
        net = _make_net(False)
        net.set_mode(3)
        net.train()
        torch.manual_seed(7)
        a = net(x)
        torch.manual_seed(7)
        b = net(x)
        self.assertEqual(tuple(a.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(a, b))
        self._assert_runtime_device(a)

    def test_inference_flag_on_token_count_and_mask_cut(self):
        # Plan: first slice dropped -> transformer input has 2*h*w*(S-1) tokens. Mask boundary mirrors the
        # single-view ``top*h*w - 1`` in token units -> dual ``top*2*h*w - 1``. Unpadded volume => top == S-1.
        h, w, s_full = 8, 8, 5
        x = _rand(1, 1, 32, 32, s_full)
        net = _make_net(True)
        net.set_mode(3)
        net.eval()
        with torch.no_grad():
            out, seen = _capture_rnn_input(net, x)
        self.assertEqual(tuple(out.shape), (1, 3, 1))
        self.assertEqual(seen['tokens'][1], 2 * h * w * (s_full - 1))
        top = s_full - 1
        self.assertEqual(int(seen['top'].reshape(-1)[0]), top * 2 * h * w - 1)
        self._assert_runtime_device(out)

    def test_inference_flag_on_mode2_shape(self):
        """TARGET RNN-only inference uses mode 2 (dual-view modes: 2, 3, -1)."""
        x = _rand(1, 1, 32, 32, 5)
        net = _make_net(True)
        net.set_mode(2)
        net.eval()
        with torch.no_grad():
            out = net(x)
        self.assertEqual(tuple(out.shape), (1, 1))
        self._assert_runtime_device(out)

    def test_flag_off_matches_master_forward_numerically(self):
        """Flag-off worktree forward_ vs untouched master create_rAIdiologist_v5_1, same seed."""
        master_root = _master_pkg_root()
        if master_root is None:
            self.skipTest('master tree not found; set RAIDIOLOGIST_MASTER_ROOT')
        x = _rand(1, 1, 32, 32, 5)
        # eval, mode 3
        net = _make_net(False)
        net.set_mode(3)
        net.eval()
        torch.manual_seed(11)
        with torch.no_grad():
            wt_eval = net(x)
        master_eval, msg_eval = _master_forward(x, 0, 11, 3, training=False)
        self.assertIsNotNone(master_eval, msg_eval)
        print(msg_eval, flush=True)
        self._assert_runtime_device(wt_eval)
        self.assertTrue(
            torch.allclose(wt_eval.detach().cpu(), master_eval, rtol=1e-5, atol=1e-6),
            'eval flag-off diverged from master forward_',
        )
        # train, mode 3 (dropout RNG aligned by seed)
        net_t = _make_net(False)
        net_t.set_mode(3)
        net_t.train()
        torch.manual_seed(13)
        wt_train = net_t(x)
        master_train, msg_train = _master_forward(x, 0, 13, 3, training=True)
        self.assertIsNotNone(master_train, msg_train)
        print(msg_train, flush=True)
        self._assert_runtime_device(wt_train)
        self.assertTrue(
            torch.allclose(wt_train.detach().cpu(), master_train, rtol=1e-5, atol=1e-6),
            'train flag-off diverged from master forward_',
        )


if __name__ == '__main__':
    unittest.main()
