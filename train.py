import torchvision
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet50
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # 指定设备
print(device)

# 定义数据处理                         transforms： 由transform构成的列表
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),        # 先随机切，再resize到给定大小
                                 transforms.RandomHorizontalFlip(),        # 水平翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),    # 正则化， 给定均值（R, G, B），方差（R, G, B）
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),       # 进行中心切割
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }


"""
# 加载图片
file = 'flower_data/'
train_file = file + 'train'           # 'flower_data/train'
val_file = file + 'val'               # 'flower_data/val'
class_name = os.listdir(train_file)     # 种类名称
class_dict = {}               # 定义种类名称和编号
for i in range(len(class_name)):        # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    class_dict[class_name[i]] = i

X_train = []
Y_train = []
for cla in class_name:
    class_path = train_file + '/' + cla     # 'flower_data/train/cla'
    y = class_dict[cla]           # 种类编号
    train_image = os.listdir(class_path)
    for f in train_image:
        path = class_path + '/' + f        # 'flower_data/train/cla/图片名称'
        img = Image.open(path)             # 打开图片
        X_train.append(img)
        Y_train.append(y)

X_val = []
Y_val = []
for cla in class_name:
    class_path = val_file + '/' + cla     # 'flower_data/val/cla'
    y = class_dict[cla]           # 种类编号
    val_image = os.listdir(class_path)
    for f in val_image:
        path = class_path + '/' + f        # 'flower_data/val/cla/图片名称'
        img = Image.open(path)             # 打开图片
        X_val.append(img)
        Y_val.append(y)
"""


image_path = "flower_data/"
train_dataset = datasets.ImageFolder(root=image_path+"train",                 # 返回一个类对象，从路径中读取图像数据，并经过transform变换
                                     transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root=image_path+"val",
                                   transform=data_transform["val"])

train_num = len(train_dataset)           # 训练集数量3306
val_num = len(val_dataset)               # 验证集数量
flower_list = train_dataset.class_to_idx    # 类名对应的索引 {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
f = {}
for name, idx in enumerate(flower_list):
    f[str(name)] = idx
flower_list = f

# 把种类dict写入json文件中
json_str = json.dumps(flower_list)
with open('class_data.json', 'w') as json_file:
    json_file.write(json_str)

# 加载数据为Tensor，用于之后的模型输入
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size, shuffle=False, num_workers=0)

# 获取网络
#net = resnet34(num_classes=5)
net = resnet50(num_classes=5)

net.to(device)                            # 模型部署到设备上

loss_function = nn.CrossEntropyLoss()      # 定义损失为交叉熵Loss
optimizer = optim.Adam(net.parameters(), lr=0.0001)      # 定义优化器

best_acc = 0.0
#save_path = './resNet34.pth'       # 保存权重路径
save_path = './resNet50.pth'
for epoch in range(3):
    # 开始训练
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data         # 包含了一个batch_size的样本
        optimizer.zero_grad()
        logits = net(images.to(device))     # 模型输出
        loss = loss_function(logits, labels.to(device))   # 计算loss
        loss.backward()          # 反向传播
        optimizer.step()         # 更新

        running_loss += loss.item()
        rate = (step+1)/len(train_loader)      # 进度
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain_loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

        val_accurate = acc / val_num       # 计算准确率
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print("完成训练！")




















