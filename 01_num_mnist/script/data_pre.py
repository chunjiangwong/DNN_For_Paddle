import paddle as paddle

def read_data(buf_size,batch_size):

    # 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(),
                              buf_size=buf_size),
        batch_size=batch_size)
    # 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
    test_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.test(),
                              buf_size=buf_size),
        batch_size=batch_size)

    # 用于打印，查看mnist数据
    train_data = paddle.dataset.mnist.train()
    sampledata = next(train_data())
    print(sampledata)
    return train_reader, test_reader