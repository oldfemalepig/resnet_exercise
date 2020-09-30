# resnet_exercise

## 文件介绍
flower_data：放置数据集，其中train和val是划分数据集后产生的<br>
test：放置测试的图片<br>
model.py：模型文件<br>
predict.py：进行预测<br>
split_data.py：对数据集进行划分<br>
train.py：训练网络<br>
class_data.json：编号-种类的对应

## 环境介绍
pytorch：1.6.0<br>
torchvision：0.7.0

## 过程
### 一、下载数据集
1、 打开`/flower_data/flower_link.txt`中的链接，下载数据集<br>
2、随后运行`split_data.py`，划分训练集和验证集
### 二、搭建模型
1、构建BasicBlock<br>
先定义结构，由两块组成；再定义前向传播函数<br>
2、构建Bottleneck<br>
同样先定义结构，由三块组成（降维、卷积、升维）；再定义前向传播函数<br>
3、构建ResNet类<br>
1）先定义第一阶段，为一个7x7的卷积处理，stride为2，然后经过池化处理<br>
2）定义4个block，这里采用了_make_layer()函数来产生<br>
3）最后采用平均池化，定义全连接层。<br>
4）定义前向传播函数

### 三、训练模型
1、定义数据处理<br>
训练集：先随机切割再resize到224大小，水平翻转，转化为Tensor，最后正则化<br>
验证集：先resize到256，再进行中心切割，转化为Tensor，最后正则化<br>
2、加载图片<br>
采用`datasets.ImageFolder`从路径中读取图像数据，并经过transforms变换<br>
构建种类-编号的dict，并写入json文件中<br>
采用`torch.utils.data.DataLoader`，按照一个bacth_size，分批次加载数据为Tensor类型<br>
3、获取网络<br>
定义Loss和优化器<br>
开始迭代训练

### 四、预测
可更改预测图片路径，来完成对图片的预测输出
