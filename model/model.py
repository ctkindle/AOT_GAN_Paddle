#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm

#  Aggregated Contextual Transformations模块，是构成模型的核心结构，负责提取多尺度特征
class AOTBlock(nn.Layer):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()

        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.Pad2D(rate, mode='reflect'),
                    nn.Conv2D(dim, dim//4, 3, 1, 0, dilation=int(rate)),
                    nn.ReLU()))
        self.fuse = nn.Sequential(
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))
        self.gate = nn.Sequential(
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = paddle.concat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = F.sigmoid(mask)
        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class UpConv(nn.Layer):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2D(inc, outc, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

# 生成器
class InpaintGenerator(nn.Layer):
    def __init__(self, opt):
        super(InpaintGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Pad2D(3, mode='reflect'),
            nn.Conv2D(4, 64, 7, 1, 0),
            nn.ReLU(),
            nn.Conv2D(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2D(128, 256, 4, 2, 1),
            nn.ReLU()
        )

        self.middle = nn.Sequential(*[AOTBlock(256, opt.rates) for _ in range(opt.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(),
            UpConv(128, 64),
            nn.ReLU(),
            nn.Conv2D(64, 3, 3, 1, 1)
        )

    def forward(self, x, mask):
        x = paddle.concat([x, mask], 1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = paddle.tanh(x)

        return x

# 判别器
class Discriminator(nn.Layer):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2D(inc, 64, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(64, 128, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(128, 256, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(256, 512, 4, 1, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2D(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        return feat

# 用于计算Perceptual Loss和Style Loss的vgg19模型（使用ImageNet预训练权重）
class VGG19F(nn.Layer):
    def __init__(self):
        super(VGG19F, self).__init__()

        self.feature_0 = nn.Conv2D(3, 64, 3, 1, 1)
        self.relu_1 = nn.ReLU()
        self.feature_2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.relu_3 = nn.ReLU()

        self.mp_4 = nn.MaxPool2D(2, 2, 0)
        self.feature_5 = nn.Conv2D(64, 128, 3, 1, 1)
        self.relu_6 = nn.ReLU()
        self.feature_7 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu_8 = nn.ReLU()

        self.mp_9 = nn.MaxPool2D(2, 2, 0)
        self.feature_10 = nn.Conv2D(128, 256, 3, 1, 1)
        self.relu_11 = nn.ReLU()
        self.feature_12 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_13 = nn.ReLU()
        self.feature_14 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_15 = nn.ReLU()
        self.feature_16 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_17 = nn.ReLU()

        self.mp_18 = nn.MaxPool2D(2, 2, 0)
        self.feature_19 = nn.Conv2D(256, 512, 3, 1, 1)
        self.relu_20 = nn.ReLU()
        self.feature_21 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_22 = nn.ReLU()
        self.feature_23 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_24 = nn.ReLU()
        self.feature_25 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_26 = nn.ReLU()

        self.mp_27 = nn.MaxPool2D(2, 2, 0)
        self.feature_28 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_29 = nn.ReLU()
        self.feature_30 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_31 = nn.ReLU()
        self.feature_32 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_33 = nn.ReLU()
        self.feature_34 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_35 = nn.ReLU()

    def forward(self, x):
        x = self.stand(x)
        feats = []
        group = []
        x = self.feature_0(x)
        x = self.relu_1(x)
        group.append(x)
        x = self.feature_2(x)
        x = self.relu_3(x)
        group.append(x)
        feats.append(group)
        
        group = []
        x = self.mp_4(x)
        x = self.feature_5(x)
        x = self.relu_6(x)
        group.append(x)
        x = self.feature_7(x)
        x = self.relu_8(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_9(x)
        x = self.feature_10(x)
        x = self.relu_11(x)
        group.append(x)
        x = self.feature_12(x)
        x = self.relu_13(x)
        group.append(x)
        x = self.feature_14(x)
        x = self.relu_15(x)
        group.append(x)
        x = self.feature_16(x)
        x = self.relu_17(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_18(x)
        x = self.feature_19(x)
        x = self.relu_20(x)
        group.append(x)
        x = self.feature_21(x)
        x = self.relu_22(x)
        group.append(x)
        x = self.feature_23(x)
        x = self.relu_24(x)
        group.append(x)
        x = self.feature_25(x)
        x = self.relu_26(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_27(x)
        x = self.feature_28(x)
        x = self.relu_29(x)
        group.append(x)
        x = self.feature_30(x)
        x = self.relu_31(x)
        group.append(x)
        x = self.feature_32(x)
        x = self.relu_33(x)
        group.append(x)
        x = self.feature_34(x)
        x = self.relu_35(x)
        group.append(x)
        feats.append(group)

        return feats

    def stand(self, x):
        mean = paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        std = paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        y = (x + 1.) / 2.
        y = (y - mean) / std
        return y
