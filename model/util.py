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

import os
import numpy as np

# 初始化输出文件夹（默认为项目路径下的output/文件夹
def init_output(output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        # 记录当前迭代步数
        current_step = np.array([0])
        np.save(os.path.join(output_path, "current_step"), current_step)
        print('训练输出目录['+output_path+']初始化完成')
    # 存储生成器、判别器check point
    if not os.path.exists(os.path.join(output_path, "model")):
        os.mkdir(os.path.join(output_path, "model"))
    # 存储训练时生成的图片
    if not os.path.exists(os.path.join(output_path, 'pic')):
        os.mkdir(os.path.join(output_path, 'pic'))
    # 存储预测时生成的图片
    if not os.path.exists(os.path.join(output_path, 'pic_val')):
        os.mkdir(os.path.join(output_path, 'pic_val'))

