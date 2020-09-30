"""
预测函数
"""
import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # 指定设备


# 定义数据变换
data_transform = transforms.Compose(
    [transforms.Resize(256),       # resize到256
    transforms.CenterCrop(224),    # 中心裁剪224
    transforms.ToTensor(),        # 转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]    # 正则化
)

img = Image.open("test/3.jpg")    # 预测图片
plt.imshow(img)

img = data_transform(img)       # 对图像进行处理, [N, C, H, W]
img = torch.unsqueeze(img, dim=0)     # torch.unsqueeze()对数据维度进行扩充， 对指定维度扩充   [1, N, C, H, W]    因为后面模型输入是这个shape


# 读取种类字典
try:
    json_file = open('./class_data.json', 'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


# 创建模型
model = resnet34(num_classes=5)
#model_weight_path = './resNet34.pth'
model_weight_path = './resNet50.pth'
model.load_state_dict(torch.load(model_weight_path, map_location=device))    # 加载模型
model.eval()
with torch.no_grad():
    output = model(img)                         # 模型输出
    output = torch.squeeze(output)              # 去掉维度
    predict = torch.softmax(output, dim=0)      # softmax输出
    predict_cla = torch.argmax(predict).numpy()

print(class_dict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()



























