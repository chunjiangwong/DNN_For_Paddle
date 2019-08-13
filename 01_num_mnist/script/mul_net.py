import paddle.fluid as fluid

"""
# # **Step2.网络配置**
# 以下的代码判断就是定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，
因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，
所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。

# 定义多层感知器
# 第一个全连接层，激活函数为ReLU
# 第二个全连接层，激活函数为ReLU
#  以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
"""
def multilayer_perceptron(input):

    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')

    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')

    prediction = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return prediction
