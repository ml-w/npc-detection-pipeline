import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_med_imaging.networks.UNet import UNet, down, up
import torchio as tio

class UNetLocTexHistDeeper(UNet):
    def __init__(self, *args, **kwargs):
        try:
            fc_inchan = kwargs.pop('fc_inchan')
        except:
            fc_inchan = 104 # 100 bins, 3 Cart Coord, 1 distance to center
        self._save_inter_res = kwargs.pop('inter_res') if 'inter_res' in kwargs else False
        self.inter_res = {}
        super(UNetLocTexHistDeeper, self).__init__(*args, **kwargs)

        self.fc = nn.Sequential(
            nn.Linear(fc_inchan, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.ReLU(inplace=True)
        )

        self.down4 = down(512, 1024, max_pool=True)
        self.down5 = down(1024, 1024, max_pool=True)

        self.up0 = up(2048, 512, True)

        self.fc6 = nn.Linear(600, 1024)
        self.fc5 = nn.Linear(600, 1024)
        self.fc4 = nn.Linear(600, 512)
        self.fc3 = nn.Linear(600, 256)
        self.fc2 = nn.Linear(600, 128)

        self.outc = nn.Conv2d(64, 2, 1)
        self.dropout1 = nn.Dropout2d(0.2, inplace=False)
        self.dropout2 = nn.Dropout2d(0.3, inplace=False)


    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        r"""Input (B × C × H × W × 1)
        """
        if self._in_chan == 1:
            x = x.squeeze().unsqueeze(1)
        else:
            x = x.squeeze()
        x1 = self.inc(x)        # 128
        x2 = self.down1(x1)     # 64
        x3 = self.down2(x2)     # 32
        x4 = self.down3(x3)     # 16
        x5 = self.down4(x4)     # 8
        x6 = self.down5(x5)     # 4
        # expand pos
        pos = self.fc(pos)

        if self._save_inter_res:
            self.inter_res['before'] = [x2, x3, x4, x5, x6]

        X = []
        for _x, _fc in zip([x2, x3, x4, x5, x6], [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]):
            _pos = _fc(pos).expand_as(_x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            _x = _x * F.relu(_pos, False)
            X.append(_x)
        x2, x3, x4, x5, x6 = X

        if self._save_inter_res:
            self.inter_res['after'] = [x2, x3, x4, x5, x6]

        x = self.up0(self.dropout2(x6), self.dropout1(x5))
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def _forward(self, subject: tio.Subject):

        pass