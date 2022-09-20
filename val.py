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

import numpy as np
import time
import os
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from model.util import init_output

import paddle
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler
from paddle.io import Dataset, DataLoader

from model.config import opt
from model.data import PlaceDateset
from model.model import InpaintGenerator, Discriminator, VGG19F
from model.loss import L1, Perceptual, Style, Adversal

def val(show_interval=1):
    # 初始化训练输出路径
    init_output(opt.output_path)
    # 读取当前训练进度
    current_step = np.load(os.path.join(opt.output_path, 'current_step.npy'))[0]
    print('已经完成 ['+str(current_step)+'] 步训练，开始验证...')

    # 设置读取数据的dataloader
    pds = PlaceDateset(opt, istrain=False)
    loader = DataLoader(pds, shuffle=False, batch_size=1, drop_last=False, num_workers=1, use_shared_memory=False)
    data_total_num = pds.__len__()

    # 初始化生成器，读取参数
    g = InpaintGenerator(opt)
    g.eval()
    time.sleep(.1)
    para = paddle.load(os.path.join(opt.output_path, "model/g.pdparams"))
    time.sleep(.1)
    g.set_state_dict(para)

    # 遍历验证集数据进行预测，并计算psnr，ssim指标
    start = time.time()
    psnr_list = []
    ssim_list = []
    pic_path = os.path.join(opt.output_path, 'pic_val')
    for step, data in enumerate(loader):
        current_step += 1

        img, mask, fname = data
        img_masked = (img * (1 - mask)) + mask

        pred_img = g(img_masked, mask)
        comp_img = (1 - mask) * img + mask * pred_img

        img_show1 = (img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
        img_show1 = (img_show1 * 255).astype('uint8')
        img_show2 = (comp_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
        img_show2 = (img_show2 * 255).astype('uint8')
        img_show3 = (comp_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
        img_show4 = mask.numpy()[0][0]
        psnr = compare_psnr(img_show1, img_show2)
        ssim = compare_ssim(img_show1, img_show2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        f = open(os.path.join(opt.output_path, 'psnr_ssim.txt'), 'a')
        logtxt = str(step) + '\t' + str(psnr) + '\t' + str(ssim) + '\r\n'
        f.write(logtxt)
        f.close()    
        # show img
        if step % show_interval == 0:
            print('current_step:', step, \
                'filename:', fname[0], \
                'psnr:', psnr, \
                'ssim:', ssim, \
                'psnr mean:', np.array(psnr_list).mean(), \
                'ssim mean:', np.array(ssim_list).mean(), \
                time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # plt.figure(figsize=(12,4),dpi=80)
            # plt.subplot(1, 4, 1)
            # plt.imshow(img_show1)
            # plt.subplot(1, 4, 2)
            # plt.imshow(img_show2)
            # plt.subplot(1, 4, 3)
            # plt.imshow(img_show3)
            # plt.subplot(1, 4, 4)
            # plt.imshow(img_show4)
            # plt.show()
            cv2.imwrite(os.path.join(pic_path, os.path.split(fname[0])[1]), cv2.cvtColor(img_show2, cv2.COLOR_BGR2RGB))
            img_show4 = (mask.numpy()[0][0] * 255).astype('uint8')
            cv2.imwrite(os.path.join(pic_path, os.path.split(fname[0])[1].replace('.', '_mask.')), img_show4)

    # 打印预测图片的psnr， ssim均值
    print('psnr mean:', np.array(psnr_list).mean(), 'ssim mean:', np.array(ssim_list).mean())

# 参数为log屏幕打印间隔，此参数不影响最终psnr和ssim指标的统计。
# 无论屏幕打印间隔设为多少，验证过程都会计算每张图片的psnr和ssim指标，以统计均值。
if __name__ == '__main__':
    # dist.spawn(val, args=(100))
    val(100)
