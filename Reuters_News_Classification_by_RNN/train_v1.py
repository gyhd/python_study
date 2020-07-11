import datetime
import numpy as np
from routing_net.RNN_RoutingMask_Model import *

# 定义参数
num_words = 10000
maxlen = 80
BATCH_SIZE = 500  # 批次
learning_rate = 1e-3  # 0.001
logs_path = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Tensorboard日志保存

# 加载数据路透社的11,228条新闻，分为了46个主题
print('Loading data...')
(x_train, y_train), _ = tf.keras.datasets.reuters.load_data(path='./reuters.npz', num_words=num_words)
# 数据对齐
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
# x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
print('Pad sequences x_train shape:', x_train.shape)
# 获取长度
x_length = np.count_nonzero(x_train, axis=1)
# x_test_length = np.count_nonzero(x_test, axis=1)
# 定义数据集
dataset = tf.data.Dataset.from_tensor_slices(((x_train, x_length), y_train)).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# 模型输出
x, x_len, y, pred = Model()

# 损失函数及优化器
cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=y), name='Loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# 准确率
correct_prediction = tf.equal(tf.cast(tf.argmax(pred, -1), tf.int64), y)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
accuracy, acc_op = tf.metrics.accuracy(labels=tf.cast(tf.argmax(pred, -1), tf.int64), predictions=y)

# 集合评估指标
tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)
merged = tf.summary.merge_all()

# 生成数据迭代器
iterator1 = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
one_element1 = iterator1.get_next()  # 获取一个元素

# 训练网络
with tf.Session() as sess:
    sess.run(iterator1.make_initializer(dataset))  # 初始化迭代器
    sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # 创建写对象
    EPOCHS = 18
    for epoch in range(EPOCHS):
        alloss = []  # 数据集迭代两次
        allacc = []
        while True:  # 通过for循环打印所有的数据
            try:
                inp, target = sess.run(one_element1)
                _, summary, acc, _, loss = sess.run([train_op, merged, accuracy, acc_op, cost],
                                                    feed_dict={x: inp[0], x_len: inp[1], y: target})
                alloss.append(loss)

                summary_writer.add_summary(summary, epoch)

            except tf.errors.OutOfRangeError:
                pass
                # print("遍历结束")
                print("step", epoch + 1, ": loss=", np.mean(alloss), "acc=", acc)
                sess.run(iterator1.make_initializer(dataset))  # 从头再来一遍
                break

    saver = tf.train.Saver()
    save_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saver.save(sess, 'train_model\\' + save_time)
