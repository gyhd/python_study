# coding:utf-8
# YOLO实现
import tensorflow as tf
from src import module
import numpy as np
slim = tf.contrib.slim

class YOLO():
    def __init__(self,class_num, anchors, width=608, height=608):
        self.class_num = class_num
        self.anchors = np.asarray(anchors).reshape([-1, 3, 2])
        self.width = width
        self.height = height
        pass

    def forward(self, inputs, batch_norm_decay=0.9, weight_decay=0.0005, isTrain=True, reuse=False):
        # set batch norm params
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': isTrain,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            # darknet53 特征
            # [N, 19, 19, 512], [N, 38, 38, 256], [N, 76, 76, 128]
            route_1, route_2, route_3 = module.extraction_feature(inputs, batch_norm_params, weight_decay)
            
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with tf.variable_scope('yolo'):
                    # 计算 y1 特征
                    # [N, 76, 76, 128] => [N, 76, 76, 256]
                    net = module.conv(route_1, 256)
                    # [N, 76, 76, 256] => [N, 76, 76, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y3 = net

                    # 计算 y2 特征
                    # [N, 76, 76, 128] => [N, 38, 38, 256]
                    net = module.conv(route_1, 256, stride=2)
                    # [N, 38, 38, 512]
                    net = tf.concat([net, route_2], -1)
                    net = module.yolo_conv_block(net, 512, 2, 1)
                    route_147 = net
                    # [N, 38, 38, 256] => [N, 38, 38, 512]
                    net = module.conv(net, 512)
                    # [N, 38, 38, 512] => [N, 38, 38, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y2 = net

                    # 计算 y3 特征
                    # [N, 38, 38, 256] => [N, 19, 19, 512]
                    net = module.conv(route_147, 512, stride=2)
                    net = tf.concat([net, route_3], -1)
                    net = module.yolo_conv_block(net, 1024, 3, 0)
                    # [N, 19, 19, 1024] => [N, 19, 19, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y1 = net

        return feature_y1, feature_y2, feature_y3

    # 计算最大的 IOU, GIOU
    def IOU(self, pre_xy, pre_wh, valid_yi_true):
        '''
            pre_xy : [13, 13, 3, 2]
            pre_wh : [13, 13, 3, 2]
            batch_yi_true : [13, 13, 3, 5 + class_num] or [13, 13, 3, 4]
            但是 shape 可以在前面加一个 batch，即以下的 shape 也是可以的, 相应的 shape 也要加一个 batch_size
            pre_xy : [batch_size, 13, 13, 3, 2]
            pre_wh : [batch_size, 13, 13, 3, 2]
            batch_yi_true : [batch_size, 13, 13, 3, 5 + class_num] or [batch_size, 13, 13, 3, 4]
            return:
                iou, giou : [13, 13, 3, V], V表示V个真值
        '''

        # [13, 13, 3, 2] ==> [13, 13, 3, 1, 2]
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        # [V, 2]
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        # 相交区域左上角坐标 : [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 相交区域右下角坐标
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        # 相交区域宽高 [13, 13, 3, V, 2] == > [13, 13, 3, V, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        # 相交区域面积 : [13, 13, 3, V]
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        # 预测box面积 : [13, 13, 3, 1]
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        # 真值 box 面积 : [V]
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        # [V] ==> [1, V]
        true_area = tf.expand_dims(true_area, axis=0)
        # iou : [13, 13, 3, V]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)    # 防止除0

        # 并集区域面积 : [13, 13, 3, V, 1] ==> [13, 13, 3, V] 
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        # giou : [13, 13, 3, V]
        giou = (intersection_area+1e-10) / combine_area # 加上一个很小的数字，确保 giou 存在
        
        return iou, giou

    # 得到低iou的地方
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, use_iou=True, ignore_thresh=0.5):
        '''
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_true:[batch_size, 13, 13, 3, 5+class_num]
        use_iou:是否使用 iou 作为计算标准
        ignore_thresh:iou小于这个值就认为与真值不重合
        return: [batch_size, 13, 13, 3, 1]
        返回 iou 低于阈值的区域 mask
        '''
        # 置信度:[batch_size, 13, 13, 3, 1]
        conf_yi_true = yi_true[..., 4:5]

        # iou小的地方
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # batch_size
        N = tf.shape(yi_true)[0]
        
        def loop_cond(index, low_iou_mask):
            return tf.less(index, N)        
        def loop_body(index, low_iou_mask):
            # 一张图片的全部真值 : [13, 13, 3, class_num+5] & [13, 13, 3, 1] == > [V, class_num + 5]
            valid_yi_true = tf.boolean_mask(yi_true[index], tf.cast(conf_yi_true[index, ..., 0], tf.bool))
            # 计算 iou/ giou : [13, 13, 3, V]
            iou, giou = self.IOU(pre_xy[index], pre_wh[index], valid_yi_true)

            # [13, 13, 3]
            if use_iou:
                best_giou = tf.reduce_max(iou, axis=-1)
            else:
                best_giou = tf.reduce_max(giou, axis=-1)
            # [13, 13, 3]
            low_iou_mask_tmp = best_giou < ignore_thresh
            # [13, 13, 3, 1]
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            # 写入
            low_iou_mask = low_iou_mask.write(index, low_iou_mask_tmp)
            return index+1, low_iou_mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        # 拼接:[batch_size, 13, 13, 3, 1]
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    # 对预测值进行解码
    def __decode_feature(self, yi_pred, curr_anchors):
        '''
        yi_pred:[batch_size, 13, 13, 3 * (class_num + 5)]
        curr_anchors:[3,2], 这一层对应的 anchor, 真实值
        return:
            xy:[batch_size, 13, 13, 3, 2], 相对于原图的中心坐标
            wh:[batch_size, 13, 13, 3, 2], 相对于原图的宽高
            conf:[batch_size, 13, 13, 3, 1]
            prob:[batch_size, 13, 13, 3, class_num]
        '''
        shape = tf.shape(yi_pred) 
        shape = tf.cast(shape, tf.float32)
        # [batch_size, 13, 13, 3, class_num + 5]
        yi_pred = tf.reshape(yi_pred, [shape[0], shape[1], shape[2], 3, 5 + self.class_num])
        # 分割预测值
        # shape : [batch_size,13,13,3,2] [batch_size,13,13,3,2] [batch_size,13,13,3,1] [batch_size,13,13,3, class_num]
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, self.class_num], axis=-1)

        ''' 计算 x,y 的坐标偏移 '''
        offset_x = tf.range(shape[2], dtype=tf.float32) #宽
        offset_y = tf.range(shape[1], dtype=tf.float32) # 高
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        # 把 offset_x, offset_y 合并成一个坐标网格 [13*13, 2], 每个元素是当前坐标 (x, y)
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        # [13, 13, 1, 2]
        offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy    
        xy = xy / [shape[2], shape[1]]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [self.width, self.height]

        return xy, wh, conf, prob

    # 计算损失
    def __compute_loss_v3(self, xy, wh, conf, prob, yi_true, low_iou_mask):
        '''
        xy:[batch_size, 13, 13, 3, 2]
        wh:[batch_size, 13, 13, 3, 2]
        conf:[batch_size, 13, 13, 3, 1]
        prob:[batch_size, 13, 13, 3, class_num]
        yi_true:[batch_size, 13, 13, 3, class_num]
        low_iou_mask:[batch_size, 13, 13, 3, 1]
        return: 总的损失

        loss_total:总的损失
        xy_loss:中心坐标损失
        wh_loss:宽高损失
        conf_loss:置信度损失
        class_loss:分类损失
        '''
        # bool => float32
        low_iou_mask = tf.cast(low_iou_mask, tf.float32)
        # batch_size
        N = tf.shape(xy)[0]
        N = tf.cast(N, tf.float32)

        # [batch_size, 13, 13, 3, 1]
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = no_obj_mask * low_iou_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:,:,:,:,4:5], logits=conf)

        # [batch_size, 13, 13, 3, 1]
        obj_mask = yi_true[..., 4:5]
        conf_loss_obj = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:,:,:,:,4:5], logits=conf)
        
        # 置信度损失
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.reduce_sum(conf_loss) / N
        
        # 平衡系数
        loss_scale = tf.square(2. - yi_true[..., 2:3] * yi_true[..., 3:4])

        # xy 损失
        xy_loss = loss_scale * obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.reduce_sum(xy_loss) / N

        # wh 损失
        wh_y_true = tf.where(condition=tf.equal(yi_true[..., 2:4], 0),
                                        x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.equal(wh, 0),
                                        x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = loss_scale * obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.reduce_sum(wh_loss) / N

        # prob 损失
        class_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[...,5:5+self.class_num], logits=prob)
        class_loss = tf.reduce_sum(class_loss) / N

        loss_total = xy_loss + wh_loss + conf_loss + class_loss
        return loss_total
        
    # 获得损失
    def get_loss(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, use_iou=True, ignore_thresh=0.5):
        '''
        feature_y1:[batch_size, 13, 13, 3*(5+class_num)]
        feature_y2:[batch_size, 26, 26, 3*(5+class_num)]
        feature_y3:[batch_size, 52, 52, 3*(5+class_num)]
        return : loss_total, loss_xy, loss_wh, loss_conf, loss_class
        use_iou:bool, 使用 iou 作为与真值匹配的衡量标准, 否则使用 giou 
        ignore_thresh    # iou低于这个数就认为与真值不重合
        return:total_loss
        '''
        # y1
        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.anchors[2])
        low_iou_mask_y1 = self.__get_low_iou_mask(xy, wh, y1_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y1 = self.__compute_loss_v3(xy, wh, conf, prob, y1_true, low_iou_mask_y1)

        # y2
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.anchors[1])
        low_iou_mask_y2 = self.__get_low_iou_mask(xy, wh, y2_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y2 = self.__compute_loss_v3(xy, wh, conf, prob, y2_true, low_iou_mask_y2)

        # y3
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.anchors[0])
        low_iou_mask_y3 = self.__get_low_iou_mask(xy, wh, y3_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y3 = self.__compute_loss_v3(xy, wh, conf, prob, y3_true, low_iou_mask_y3)

        return loss_y1 + loss_y2 + loss_y3

    # 非极大值抑制
    def __nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_threshold=0.5):
        '''
        boxes:[1, V, 4]
        score:[1, V, class_num]
        num_classes:分类数
        max_boxes:一类最大保留多少个 box
        score_thresh:小于这个分数的不算
        iou_threshold:iou大于这个的合并
        return:????
            boxes:[V, 4]
            score:[V,]
        '''
        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # [V, 4]
        boxes = tf.reshape(boxes, [-1, 4])
        # [V, class_num]
        score = tf.reshape(scores, [-1, num_classes])

        # 分数大的掩码
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # 对每一个分类进行 nms 操作
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:,i])
            filter_score = tf.boolean_mask(score[:,i], mask[:,i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_score,
                                                    max_output_size=max_boxes,
                                                    iou_threshold=iou_threshold, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        # 合并
        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label

    # 得到预测的全部 box
    def __get_pred_box(self, feature_y1, feature_y2, feature_y3):
        '''
        feature_y1:[1, 13, 13, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        return:
            boxes:[1, V, 4]:[x_min, y_min, x_max, y_max] 相对于原始图片大小的浮点数
            conf:[1, V, 1]
            prob:[1, V, class_num]
        '''
        # y1解码
        xy1, wh1, conf1, prob1 = self.__decode_feature(feature_y1, self.anchors[2])
        conf1, prob1 = tf.sigmoid(conf1), tf.sigmoid(prob1)

        # y2解码
        xy2, wh2, conf2, prob2 = self.__decode_feature(feature_y2, self.anchors[1])
        conf2, prob2 = tf.sigmoid(conf2), tf.sigmoid(prob2)

        # y3解码
        xy3, wh3, conf3, prob3 = self.__decode_feature(feature_y3, self.anchors[0])
        conf3, prob3 = tf.sigmoid(conf3), tf.sigmoid(prob3)

        # 把 box 放到一起来
        def _reshape(xy, wh, conf, prob):
            # [1, 13, 13, 3, 1]
            x_min = xy[..., 0: 1] - wh[..., 0: 1] / 2.0
            x_max = xy[..., 0: 1] + wh[..., 0: 1] / 2.0
            y_min = xy[..., 1: 2] - wh[..., 1: 2] / 2.0
            y_max = xy[..., 1: 2] + wh[..., 1: 2] / 2.0

            # [1, 13, 13, 3, 4]
            boxes = tf.concat([x_min, y_min, x_max, y_max], -1)
            shape = tf.shape(boxes)
            # [1, 13*13*3, 4]
            boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2]* shape[3], shape[4]))

            # [1, 13 * 13 * 3, 1]
            conf = tf.reshape(conf, (shape[0], shape[1] * shape[2]* shape[3], 1))

            # [1, 13 * 13 * 3, class_num]
            prob = tf.reshape(prob, (shape[0], shape[1] * shape[2]* shape[3], -1))
        
            return boxes, conf, prob

        # reshape
        # [batch_size, 13*13*3, 4], [batch_size, 13*13*3, 1], [batch_size, 13*13*3, class_num]
        boxes_y1, conf_y1, prob_y1 = _reshape(xy1, wh1, conf1, prob1)
        boxes_y2, conf_y2, prob_y2 = _reshape(xy2, wh2, conf2, prob2)
        boxes_y3, conf_y3, prob_y3 = _reshape(xy3, wh3, conf3, prob3)

        # 全部拿到一起来
        # [1, 13*13*3, 4] & [1, 26*26*3, 4] & [1, 52*52*3, 4] ==> [1,  V, 4]
        boxes = tf.concat([boxes_y1, boxes_y2, boxes_y3], 1)
        conf = tf.concat([conf_y1, conf_y2, conf_y3], 1)
        prob = tf.concat([prob_y1, prob_y2, prob_y3], 1)

        return boxes, conf, prob

    # 得到预测结果
    def get_predict_result(self, feature_y1, feature_y2, feature_y3, class_num, score_thresh=0.5, iou_thresh=0.5, max_box=200):
        '''
        feature_y1:[batch_size, 13, 13, 3*(class_num+5)]
        feature_y2:[batch_size, 26, 26, 3*(class_num+5)]
        feature_y3:[batch_size, 52, 52, 3*(class_num+5)]
        class_num:分类数
        score_thresh:小于这个分数的就不算
        iou_thresh : 超过这个 iou 的 box 进行合并
        max_box : 最多保留多少物体
        return:
            boxes:[V, 4]包含[x_min, y_min, x_max, y_max]
            score:[V, 1]
            label:[V, 1]
        '''
        boxes, conf, prob = self.__get_pred_box(feature_y1, feature_y2, feature_y3)
        pre_score = conf * prob
        boxes, score, label = self.__nms(boxes, pre_score, class_num, max_boxes=max_box, score_thresh=score_thresh, iou_threshold=iou_thresh)
        return boxes, score, label

