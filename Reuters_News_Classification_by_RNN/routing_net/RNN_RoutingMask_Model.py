import tensorflow as tf


def Model():
    def mkMask(input_tensor, maxLen):  # 计算变长RNN的掩码
        shape_of_input = tf.shape(input_tensor)
        shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

        oneDtensor = tf.reshape(input_tensor, shape=(-1,))
        flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
        return tf.reshape(flat_mask, shape_of_output)

    # 定义函数，将输入转化成uhat
    def shared_routing_uhat(caps,  # 输入 shape(b_sz, maxlen, caps_dim)
                            out_caps_num,  # 输出胶囊个数
                            out_caps_dim, scope=None):  # 输出胶囊维度

        batch_size, maxlen = tf.shape(caps)[0], tf.shape(caps)[1]  # 获取批次和长度

        with tf.variable_scope(scope or 'shared_routing_uhat'):  # 转成uhat
            caps_uhat = tf.layers.dense(caps, out_caps_num * out_caps_dim, activation=tf.nn.tanh)
            caps_uhat = tf.reshape(caps_uhat, shape=[batch_size, maxlen, out_caps_num, out_caps_dim])

        return caps_uhat  # 输出batch_size, maxlen, out_caps_num, out_caps_dim

    def masked_routing_iter(caps_uhat, seqLen, iter_num):  # 动态路由计算

        assert iter_num > 0
        batch_size, maxlen = tf.shape(caps_uhat)[0], tf.shape(caps_uhat)[1]  # 获取批次和长度
        out_caps_num = int(caps_uhat.get_shape()[2])
        seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
        mask = mkMask(seqLen, maxlen)  # shape(batch_size, maxlen)
        floatmask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)  # shape(batch_size, maxlen, 1)

        # shape(b_sz, maxlen, out_caps_num)
        B = tf.zeros([batch_size, maxlen, out_caps_num], dtype=tf.float32)
        for i in range(iter_num):
            C = tf.nn.softmax(B, axis=2)  # shape(batch_size, maxlen, out_caps_num)
            C = tf.expand_dims(C * floatmask, axis=-1)  # shape(batch_size, maxlen, out_caps_num, 1)
            weighted_uhat = C * caps_uhat  # shape(batch_size, maxlen, out_caps_num, out_caps_dim)

            S = tf.math.reduce_sum(weighted_uhat, axis=1)  # shape(batch_size, out_caps_num, out_caps_dim)

            V = _squash(S, axes=[2])  # shape(batch_size, out_caps_num, out_caps_dim)
            V = tf.expand_dims(V, axis=1)  # shape(batch_size, 1, out_caps_num, out_caps_dim)
            B = tf.math.reduce_sum(caps_uhat * V, axis=-1) + B  # shape(batch_size, maxlen, out_caps_num)

        V_ret = tf.squeeze(V, axis=[1])  # shape(batch_size, out_caps_num, out_caps_dim)
        S_ret = S
        return V_ret, S_ret

    def _squash(in_caps, axes):  # 定义_squash激活函数
        _EPSILON = 1e-9
        vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSILON)
        vec_squashed = scalar_factor * in_caps  # element-wise
        return vec_squashed

    # 定义函数，使用动态路由对RNN结果信息聚合
    def routing_masked(in_x, xLen, out_caps_dim, out_caps_num, iter_num=3,
                       dropout=None, is_train=False, scope=None):
        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
        b_sz = tf.shape(in_x)[0]
        with tf.variable_scope(scope or 'routing'):
            caps_uhat = shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope='rnn_caps_uhat')
            attn_ctx, S = masked_routing_iter(caps_uhat, xLen, iter_num)
            attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num * out_caps_dim])
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    # 模型搭建
    # 参数设定
    num_words = 10000  # 词汇量限定
    maxlen = 80  # 每句话的词量控制
    nb_features = 128  # 词嵌入维度
    out_caps_num = 5  # 定义输出的胶囊个数
    n_classes = 46  # 分类个数

    x = tf.placeholder("float", [None, maxlen], name='input_word')  # 定义输入占位符
    x_len = tf.placeholder(tf.int32, [None, ], name="input_len")  # 定义输入序列长度占位符
    y = tf.placeholder(tf.int64, [None, ], name="label")  # 定义输入分类标签占位符

    with tf.name_scope('embeddings'):
        embeddings = tf.keras.layers.Embedding(num_words, nb_features)(x)
    with tf.name_scope('rnncells'):
        # 定义带有IndyLSTMCell的RNN网络
        hidden = [100, 50, 30]  # RNN单元个数
        stacked_rnn = []
        for i in range(3):
            cell = tf.contrib.rnn.IndyLSTMCell(hidden[i])
            stacked_rnn.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8))
        mcell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn)
    with tf.name_scope('rnnoutputs'):
        rnnoutputs, _ = tf.nn.dynamic_rnn(mcell, embeddings, dtype=tf.float32)
    with tf.name_scope('routing_masked'):
        outputs = routing_masked(rnnoutputs, x_len, int(rnnoutputs.get_shape()[-1]), out_caps_num, iter_num=3)

    print(outputs.get_shape())

    return x, x_len, y, tf.layers.dense(outputs, n_classes, activation=tf.nn.relu)
