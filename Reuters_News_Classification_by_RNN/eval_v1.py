import tensorflow as tf
import numpy as np

num_words = 10000
maxlen = 80
_, (x_test, y_test) = tf.keras.datasets.reuters.load_data(path='./reuters.npz', num_words=num_words)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
x_test_length = np.count_nonzero(x_test, axis=1)
# 完成训练后直接进行评估
ckpt = tf.train.get_checkpoint_state('train_model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
# 读取后直接创建回话
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    # 初始化图模型
    graph = tf.get_default_graph()
    # 加载操作节点
    input_word = graph.get_operation_by_name('input_word').outputs[0]
    input_len = graph.get_operation_by_name('input_len').outputs[0]
    label = graph.get_operation_by_name('label').outputs[0]
    # 测试数据
    in_data = x_test
    length = x_test_length
    true_label = y_test
    feed_dict = {
        input_word.name: in_data,
        input_len.name: length,
        label.name: true_label
    }
    # 计算
    loss = sess.run('Loss:0', feed_dict)
    accuracy = sess.run('Accuracy:0', feed_dict)
    print('Evaluation Result')
    print('--------------------')
    print('Loss:', loss)
    print('Accuracy:', accuracy)
