"""
划分数据集
"""


import os
from shutil import copy
import random
from PIL import Image


def mkfile(file):
    if not os.path.exists(file):
        os.mkdir(file)

file = 'flower_data/flower_photos'

# 设立训练集
mkfile('flower_data/train')
mkfile('flower_data/val')
flower_class = []
for filename in os.listdir("flower_data/flower_photos"):     # 列举出该目录下的所有文件
    if ".txt" not in filename:                               # 不要那个txt的文件
        flower_class.append(filename)                        # 剩下的文件名就是花的种类名

for cla in flower_class:
    mkfile('flower_data/train/' + cla)           # 为每个花的种类建立训练集

# 设立验证集

for cla in flower_class:
    mkfile('flower_data/val/' + cla)            # 为每个花的种类建立验证集


split_rate = 0.1           # 划分比例为0.1

for cla in flower_class:
    cla_path = file + '/' + cla + '/'              # 'flower_data/flower_photos/cla/'
    images = os.listdir(cla_path)               # 列举出该目录下的所有图片名称
    num = len(images)                    # 该种类花的图片总数
    eval_index = random.sample(images, k=int(num * split_rate))       # 从所有图片中，随机挑选出 num * split_rate个 组成验证集
    for index, image in enumerate(images):
        if image in eval_index:                 # 如果这个图片在验证集中
            image_path = cla_path + image        # 'flower_data/flower_photos/cla/image'    图片路径
            new_path = 'flower_data/val/' + cla    # 'flower_data/val/cla'
            copy(image_path, new_path)            # 把image_path的图片  浅拷贝到 new_path中
        else:                                 # 不在验证集中
            image_path = cla_path + image
            new_path = 'flower_data/train/' + cla
            copy(image_path, new_path)

        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")
    print()
print("processing done!")















