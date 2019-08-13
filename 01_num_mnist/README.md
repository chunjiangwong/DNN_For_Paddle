##  识别数字
##  目录
- [介绍](#introduction)
- [数据预处理](#data-preparation)
- [训练](#train)
- [评价](#evaluate)
- [测试](#infer-and-visualize)
- [生成模型](#released-model)

### Introduction

[数字识别](https://arxiv.org/abs/1512.02325) 
-------------
**Step1：准备数据。**
  (1)数据集介绍
  MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。
  ![](https://ai-studio-static-online.cdn.bcebos.com/fc73217ae57f451a89badc801a903bb742e42eabd9434ecc8089efe19a66c076) 
 (2)train_reader和test_reader
 paddle.dataset.mnist.train()和test()分别用于获取mnist训练集和测试集
 paddle.reader.shuffle()表示每次缓存BUF_SIZE个数据项，并进行打乱
 paddle.batch()表示每BATCH_SIZE组成一个batch
 （3）打印看下数据是什么样的？PaddlePaddle接口提供的数据已经经过了归一化、居中等处理。

 **Step2.网络配置**
 以下的代码判断就是定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。
![](https://ai-studio-static-online.cdn.bcebos.com/cb69f928778c4299b75814179607a89eea770bdc409d4e08a87e2975cb96b19b)


# 定义多层感知器 

# #使用交叉熵损失函数,描述真实样本标签和预测概率之间的差值

定义了一个损失函数之后，还要对它求平均值，
训练程序必须返回平均损失作为第一个返回值，因为它会被后面反向传播算法所用到。
同时我们还可以定义一个准确率函数，这个可以在我们训练的时候输出分类的准确率。



