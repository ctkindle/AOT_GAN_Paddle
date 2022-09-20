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

import paddle
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler
from paddle.io import Dataset, DataLoader

from model.config import opt
from model.data import PlaceDateset
from model.model import InpaintGenerator, Discriminator, VGG19F
from model.loss import L1, Perceptual, Style, Adversal
from model.util import init_output

def train(show_interval=100, save_interval=500, total_iter=1000000, epoch_num=1000000):
    # 初始化训练输出路径
    init_output(opt.output_path)
    
    dist.init_parallel_env()

    # 读取当前训练进度
    current_step = np.load(os.path.join(opt.output_path, 'current_step.npy'))[0]
    print('已经完成 ['+str(current_step)+'] 步训练，开始继续训练...')

    # 定义数据读取用的DataLoader
    pds = PlaceDateset(opt)
    batchsamp = DistributedBatchSampler(pds, shuffle=True, batch_size=opt.batch_size, drop_last=True)
    loader = DataLoader(pds, batch_sampler=batchsamp, num_workers=4)
    data_total_num = pds.__len__()

    # 初始化生成器、判别器、计算Perceptu Loss用的VGG19模型（权重迁移自PyTorch）
    # vgg模型不参与训练，设为预测模式
    g = InpaintGenerator(opt)
    g = paddle.DataParallel(g)
    d = Discriminator()
    d = paddle.DataParallel(d)
    vgg19 = VGG19F()
    vgg_state_dict = paddle.load(opt.vgg_weight_path)
    vgg19.set_state_dict(vgg_state_dict)
    g.train()
    d.train()
    vgg19.eval()

    # 定义优化器
    opt_g = paddle.optimizer.Adam(learning_rate=opt.lrg, beta1=opt.beta1, beta2=opt.beta2, parameters=g.parameters())
    opt_d = paddle.optimizer.Adam(learning_rate=opt.lrd, beta1=opt.beta1, beta2=opt.beta2, parameters=d.parameters())

    # 读取保存的模型权重、优化器参数
    if current_step > 0:
        print('读取存储的模型权重、优化器参数...')
        time.sleep(.1)
        para = paddle.load(os.path.join(opt.output_path, "model/g.pdparams"))
        time.sleep(.1)
        g.set_state_dict(para)
        time.sleep(.1)
        para = paddle.load(os.path.join(opt.output_path, "model/d.pdparams"))
        time.sleep(.1)
        d.set_state_dict(para)
        time.sleep(.1)
        para = paddle.load(os.path.join(opt.output_path, "model/g.pdopt"))
        time.sleep(.1)
        opt_g.set_state_dict(para)
        time.sleep(.1)
        para = paddle.load(os.path.join(opt.output_path, "model/d.pdopt"))
        time.sleep(.1)
        opt_d.set_state_dict(para)
        time.sleep(.1)

    # 定义各部分loss
    l1_loss = L1()
    perceptual_loss = Perceptual(vgg19)
    style_loss = Style(vgg19)
    adv_loss = Adversal()

    # 设置训练时生成图片的存储路径
    pic_path = os.path.join(opt.output_path, 'pic')
              
    # 训练循环
    for epoch in range(epoch_num):
        start = time.time()
        if current_step >= total_iter:
            break
        for step, data in enumerate(loader):
            if current_step >= total_iter:
                break
            current_step += 1

            # 给图片加上mask
            img, mask, fname = data
            img_masked = (img * (1 - mask)) + mask
            pred_img = g(img_masked, mask)
            comp_img = (1 - mask) * img + mask * pred_img

            # 模型参数更新过程
            loss_g = {}
            loss_g['l1'] = l1_loss(img, pred_img) * opt.l1_weight
            loss_g['perceptual'] = perceptual_loss(img, pred_img) * opt.perceptual_weight
            loss_g['style'] = style_loss(img, pred_img) * opt.style_weight
            dis_loss, gen_loss = adv_loss(d, comp_img, img, mask)
            loss_g['adv_g'] = gen_loss * opt.adversal_weight
            loss_g_total = loss_g['l1'] + loss_g['perceptual'] + loss_g['style'] + loss_g['adv_g']
            loss_d_fake = dis_loss[0]
            loss_d_real = dis_loss[1]
            loss_d_total = loss_d_fake + loss_d_real
            opt_g.clear_grad()
            opt_d.clear_grad()
            loss_g_total.backward()
            loss_d_total.backward()
            opt_g.step()
            opt_d.step()

            # 写log文件，保存生成的图片，定期保存模型check point
            log_interval = 1 if current_step < 10000 else 100
            if dist.get_rank() == 0: # 只在主进程执行
                if current_step % log_interval == 0:
                    logfn = 'log.txt'
                    f = open(os.path.join(opt.output_path, logfn), 'a')
                    logtxt = 'current_step:[' + str(current_step) +                             ']\t' + 'g_l1:' + str(loss_g['l1'].numpy()) +                             '\t' + 'g_perceptual:' + str(loss_g['perceptual'].numpy()) +                             '\t' + 'g_style:' + str(loss_g['style'].numpy()) +                             '\t' + 'g_adversal:' + str(loss_g['adv_g'].numpy()) +                             '\t' + 'g_total:' + str(loss_g_total.numpy()) +                             '\t' + 'd_fake:' + str(loss_d_fake.numpy()) +                             '\t' + 'd_real:' + str(loss_d_real.numpy()) +                             '\t' + 'd_total:' + str(loss_d_total.numpy()) +                             '\t' + 'filename:[' + fname[0] +                             ']\t' + 'time:[' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ']\n'
                    f.write(logtxt)
                    f.close()    

                # show img
                if current_step % show_interval == 0:
                    print('current_step:', current_step, 'epoch:', epoch,                         'step:['+str(step)+'/'+str(math.ceil(data_total_num / opt.batch_size))+']'                         'g_l1:', loss_g['l1'].numpy(),                         'g_perceptual:', loss_g['perceptual'].numpy(),                         'g_style:', loss_g['style'].numpy(),                         'g_adversal:', loss_g['adv_g'].numpy(),                         'g_total:', loss_g_total.numpy(),                         'd_fake:', loss_d_fake.numpy(),                         'd_real:', loss_d_real.numpy(),                         'd_total:', loss_d_total.numpy(),                         'filename:', fname[0],                         time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    # img_show1 = (img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
                    # img_show2 = (pred_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
                    # img_show3 = (comp_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
                    # img_show4 = mask.numpy()[0][0]
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

                    img_show2 = (pred_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
                    img_show2 = (img_show2 * 256).astype('uint8')
                    img_show2 = cv2.cvtColor(img_show2, cv2.COLOR_RGB2BGR)
                    img_show4 = (mask.numpy()[0][0] * 255).astype('uint8')
                    cv2.imwrite(os.path.join(pic_path, os.path.split(fname[0])[1]), img_show2)
                    cv2.imwrite(os.path.join(pic_path, os.path.split(fname[0])[1].replace('.', '_mask.')), img_show4)

                # 定时存盘
                if current_step % save_interval == 0:
                    time.sleep(.1)
                    para = g.state_dict()
                    time.sleep(.1)
                    paddle.save(para, os.path.join(opt.output_path, "model/g.pdparams"))
                    time.sleep(.1)
                    para = d.state_dict()
                    time.sleep(.1)
                    paddle.save(para, os.path.join(opt.output_path, "model/d.pdparams"))
                    time.sleep(.1)
                    para = opt_g.state_dict()
                    time.sleep(.1)
                    paddle.save(para, os.path.join(opt.output_path, "model/g.pdopt"))
                    time.sleep(.1)
                    para = opt_d.state_dict()
                    time.sleep(.1)
                    paddle.save(para, os.path.join(opt.output_path, "model/d.pdopt"))
                    time.sleep(.1)
                    np.save(os.path.join(opt.output_path, 'current_step'), np.array([current_step]))
                    print('第['+str(current_step)+']步模型保存。保存路径：', os.path.join(opt.output_path, "model"))
            
            # 存储clock
            if current_step % 10 == 0:
                clock = np.array([str(current_step), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))])
                np.savetxt(os.path.join(opt.output_path, 'clock.txt'), clock, fmt='%s', delimiter='\t')

    # 训练迭代完成时保存模型参数
    time.sleep(1)
    para = g.state_dict()
    time.sleep(1)
    paddle.save(para, os.path.join(opt.output_path, "model/g.pdparams"))
    time.sleep(1)
    para = d.state_dict()
    time.sleep(1)
    paddle.save(para, os.path.join(opt.output_path, "model/d.pdparams"))
    time.sleep(1)
    para = opt_g.state_dict()
    time.sleep(1)
    paddle.save(para, os.path.join(opt.output_path, "model/g.pdopt"))
    time.sleep(1)
    para = opt_d.state_dict()
    time.sleep(1)
    paddle.save(para, os.path.join(opt.output_path, "model/d.pdopt"))
    time.sleep(1)
    np.save(os.path.join(opt.output_path, 'current_step'), np.array([current_step]))
    print('第['+str(current_step)+']步模型保存。保存路径：', os.path.join(opt.output_path, "model"))
    print('Finished training! Total Iteration:', current_step)

# 训练时自动检测是否多卡环境，如果是则多卡并行训练。
# 训练参数分别为：log屏幕输出间隔，定期保存check point间隔， 训练总迭代次数（iteration 数量）
if __name__ == '__main__':
    dist.spawn(train, args=(1, 100, 1000000))
    # train(1, 100, 1000000)
