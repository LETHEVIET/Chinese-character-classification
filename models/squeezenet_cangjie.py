"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the root logger level (DEBUG logs everything)
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        # logging.FileHandler("my_app.log"),  # Log to a file
        logging.StreamHandler()             # Log to console 
    ]
)

log = logging.getLogger(__name__)

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=27, max_length=5):

        super().__init__()
        log.info(class_num)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(64, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)

        # self.heads = [nn.AdaptiveAvgPool2d(1) for _ in range(max_length)]
        # self.avg = nn.AdaptiveAvgPool2d(1)
        self.heads = [nn.AdaptiveAvgPool2d(1) for _ in range(max_length)]
        self.maxpool = nn.MaxPool2d(2, 2)

        self.softmax = nn.Softmax(dim=1)

        self.max_length = max_length
        self.class_num = class_num

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f3 = self.maxpool(f3)
        
        f4 = self.fire4(f3)
        f5 = self.fire5(f4) + f4
        f5 = self.maxpool(f5)

        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f9 = self.fire9(f8) + f8
        c10 = self.conv10(f9)

        # x = self.avg(c10)
        # outputs = []
        # for head, feature in zip(self.heads, c10):
        #     x = head(feature)
        #     x = x.view(x.size(0), -1)
        #     outputs.append(x)

        # return outputs

        outputs = []
        for head in self.heads:
            x = head(c10)
            x = x.view(x.size(0), -1)
            x = self.softmax(x)
            outputs.append(x)

        return outputs

def squeezenet(class_num=952):
    return SqueezeNet(class_num=27, max_length=5)
