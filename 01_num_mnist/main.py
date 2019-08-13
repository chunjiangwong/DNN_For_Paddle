# 系统包
import numpy as np
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os

import time
start = time.time()

# 自定义函数
from script.data_pre import read_data
from script.mul_net import multilayer_perceptron
from script.draw_train import draw_train_process
"""
设置参数：
    BUF_SIZE:从内存读取块的大小
    BATCH_SIZE:每批次读取多少
"""
BUF_SIZE = 512
BATCH_SIZE = 128
# (1) 读取数据集
train_reader, test_reader = read_data(buf_size=BUF_SIZE,batch_size=BATCH_SIZE)

# （2）定义数据层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')#单通道，28*28像素值
label = fluid.layers.data(name='label', shape=[1], dtype='int64')          #图片标签

# (3)获取分类器
predict = multilayer_perceptron(image)


# （4）定义损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)


# （5）定义优化函数
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)  
opts = optimizer.minimize(avg_cost)


# 在上述模型配置完毕后
# fluid.Program：
# fluid.default_startup_program()
# fluid.default_main_program() 配置完成

# 参数初始化操作会被写入fluid.default_startup_program()
# fluid.default_main_program()用于获取默认或全局main program(主程序)。

# 该主程序用于训练和测试模型。
# fluid.layers 中的所有layer函数可以向 default_main_program 中添加算子和变量。
# default_main_program 是fluid的许多编程接口（API）的Program参数的缺省值。
# 例如,当用户program没有传入的时候， Executor.run() 会默认执行 default_main_program 。

# # **Step3.模型训练 and Step4.模型评估**
# （1）创建训练的Executor
# 首先定义运算场所 fluid.CPUPlace()和 fluid.CUDAPlace(0)分别表示运算场所为CPU和GPU
# Executor:接收传入的program，通过run()方法运行program。

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# （2）告知网络传入的数据分为两部分，第一部分是image值，第二部分是label值
# DataFeeder负责将数据提供器（train_reader,test_reader）返回的数据转成一种特殊的数据结构，
# 使其可以输入到Executor中。

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


# (3)展示模型训练曲线




# （4）训练并保存模型
# 训练需要有一个训练程序和一些必要参数，并构建了一个获取训练过程中测试误差的函数
# 必要参数有executor,program,reader,feeder,fetch_list。
# **executor**表示之前创建的执行器

# **program**表示执行器所执行的program，是之前创建的program，
# 如果该项参数没有给定的话则默认使用defalut_main_program

# **reader**表示读取到的数据
# **feeder**表示前向输入的变量
# **fetch_list**表示用户想得到的变量


all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []

EPOCH_NUM=2
model_save_dir = r"model"
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):                         #遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                                        feed=feeder.feed(data),              #给模型喂入数据
                                        fetch_list=[avg_cost, acc])          #fetch 误差、准确率  
        
        all_train_iter=all_train_iter+BATCH_SIZE
        all_train_iters.append(all_train_iter)
        
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])
        
        # 每200个batch打印一次信息  误差、准确率
        if batch_id % 200 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    #每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):                         #遍历test_reader
        test_cost, test_acc = exe.run(program=test_program, #执行训练程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_accs.append(test_acc[0])                                       #每个batch的准确率
        test_costs.append(test_cost[0])                                     #每个batch的误差
        
       
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))                         #每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))                            #每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    
    #保存模型
    # 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,   #保存推理model的路径
                                  ['image'],    #推理（inference）需要 feed 的数据
                                  [predict],    #保存推理（inference）结果的 Variables
                                  exe)             #executor 保存 inference model

print('训练模型保存完成！')
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")


# # **Step5.模型预测**
# （1）图片预处理
# 在预测之前，要对图像进行预处理。
# 首先进行灰度化，然后压缩图像大小为28*28，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。


def load_image(file):
    im = Image.open(file).convert('L')                        #将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    im = im.resize((28, 28), Image.ANTIALIAS)                 #resize image with high-quality 图像大小为28*28
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)#返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    print(im)
    im = im / 255.0 * 2.0 - 1.0                               #归一化到【-1~1】之间
    return im


# （2）使用Matplotlib工具显示这张图像。
infer_path='test_data/infer_3.png'
img = Image.open(infer_path)
plt.imshow(img)   #根据数组绘制图像
plt.show()        #显示图像


# (3)创建预测用的Executer

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()


# (4)开始预测
# 通过fluid.io.load_inference_model，预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。
# 加载数据并开始预测
with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载 推理model(inference model)
    [inference_program,                                            #推理Program
     feed_target_names,                                            #是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
    img = load_image(infer_path)

    results = infer_exe.run(program=inference_program,               #运行推测程序
                   feed={feed_target_names[0]: img},           #喂入要预测的img
                   fetch_list=fetch_targets)                   #得到推测结果,  
    # 获取概率最大的label
    lab = np.argsort(results)                                  #argsort函数返回的是result数组值从小到大的索引值
    print(lab)
    # [[[6 4 1 0 7 8 5 2 9 3]]] 嵌套列表 按照概率推断，
    print("该图片的预测结果的label为: %d" % lab[0][0][-1])     #-1代表读取数组中倒数第一列  

end = time.time()

print("CPU{}".format(end-start))
# print("CPU{}".format(end-start))
# GPU 18.922019720077515
# CPU 18.135095834732056