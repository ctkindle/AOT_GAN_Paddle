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

# 设置全局变量、超参
class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        # 在AI Studio上用A100单卡训练时，设置为8（bs=8）；在使用V100四卡训练时设为6（bs=6x4=24）
        # self.batch_size = 6 # V100单卡、多卡训练
        # self.batch_size = 8 # A100单卡训练
        self.batch_size = 1
        self.img_size = 512 # 生成图片尺寸
        self.rates = [1, 2, 4, 8] # 各个尺度空洞卷积的膨胀率
        self.block_num = 8 # 生成器中AOT模块的层数
        self.l1_weight = 1 # L1 Loss的加权
        self.style_weight = 250 # Style Loss的加权
        self.perceptual_weight = .1 # Perceptu Loss的加权
        self.adversal_weight = .01 # GAN Loss的加权
        self.lrg = 1e-4 # 生成器学习率
        self.lrd = 1e-4 # 判别器学习率
        self.beta1 = .5 # Adam优化器超参
        self.beta2 = .999 # Adam优化器超参

        self.dataset_path = 'dataset' # 训练、验证数据集存放路径
        self.output_path = 'output' # chenk point，log等存放路径
        self.vgg_weight_path = 'weight/vgg19feats.pdparams' # vgg19 预训练参数存放路径
        

opt = OPT()
