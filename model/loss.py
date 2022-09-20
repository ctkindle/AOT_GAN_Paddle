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

from model.config import opt

class L1(): 
    def __init__(self,):
        self.calc = nn.L1Loss()
    
    def __call__(self, x, y):
        return self.calc(x, y)

# 计算原图片和生成图片通过vgg19模型各个层输出的激活特征图的L1 Loss
class Perceptual():
    def __init__(self, vgg, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x = F.interpolate(x, (opt.img_size, opt.img_size), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (opt.img_size, opt.img_size), mode='bilinear', align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        content_loss = 0.0
        for i in range(len(self.weights)):
            content_loss += self.weights[i] * self.criterion(x_features[i][0], y_features[i][0]) # 此vgg19预训练模型无bn层，所以尝试不用rate
        return content_loss

# 通过vgg19模型，计算原图片与生成图片风格相似性的Loss
class Style():
    def __init__(self, vgg):
        super(Style, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.shape
        f = x.reshape([b, c, w * h])
        f_T = f.transpose([0, 2, 1])
        G = paddle.matmul(f, f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x = F.interpolate(x, (opt.img_size, opt.img_size), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (opt.img_size, opt.img_size), mode='bilinear', align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        style_loss = 0.0
        blocks = [2, 3, 4, 5]
        layers = [2, 4, 4, 2]
        for b, l in list(zip(blocks, layers)):
            b = b - 1
            l = l - 1
            style_loss += self.criterion(self.compute_gram(x_features[b][l]), self.compute_gram(y_features[b][l]))
        return style_loss

# 对叠加在图片上的mask边缘进行高斯模糊处理
def gaussian_blur(input, kernel_size, sigma):
    def get_gaussian_kernel(kernel_size: int, sigma: float) -> paddle.Tensor:
        def gauss_fcn(x, window_size, sigma):
            return -(x - window_size // 2)**2 / float(2 * sigma**2)
        gauss = paddle.stack([paddle.exp(paddle.to_tensor(gauss_fcn(x, kernel_size, sigma)))for x in range(kernel_size)])
        return gauss / gauss.sum()


    b, c, h, w = input.shape
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = paddle.matmul(kernel_x, kernel_y, transpose_y=True)
    kernel = kernel_2d.reshape([1, 1, ksize_x, ksize_y])
    kernel = kernel.repeat_interleave(c, 0)
    padding = [(k - 1) // 2 for k in kernel_size]
    return F.conv2d(input, kernel, padding=padding, stride=1, groups=c)

# GAN Loss，采用最小二乘Loss
class Adversal():
    def __init__(self, ksize=71): 
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, netD, fake, real, masks): 
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake  = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.shape
        b, c, ht, wt = masks.shape
        
        # 对齐判别器输出特征图与mask的尺寸
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)
        d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize), (10, 10)).detach()
        d_real_label = paddle.zeros_like(d_real)
        g_fake_label = paddle.ones_like(g_fake)

        dis_loss = [self.loss_fn(d_fake, d_fake_label).mean(), self.loss_fn(d_real, d_real_label).mean()]
        gen_loss = (self.loss_fn(g_fake, g_fake_label) * masks / paddle.mean(masks)).mean()

        return dis_loss, gen_loss
