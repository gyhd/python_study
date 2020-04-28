# coding:utf-8
# 配置文件

# ############# 基本配置 #############
class_num = 25
anchors = 19,26, 18,38, 24,39, 26,37, 28,38, 27,40, 29,40, 32,41, 40,44
model_path = "./checkpoint/"
model_name = "model"
name_file = './data/train.names'

# ############# 日志 #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# ############## 训练 ##############
train_file = './data/train.txt'
batch_size = 2
multi_scale_img = False     # 多尺度缩放图片训练
total_epoch = 300       # 一共训练多少 epoch
save_step = 1000        # 多少步保存一次

use_iou = True      # 计算损失时, 以iou作为衡量标准, 否则用 giou
ignore_thresh = 0.5     # 与真值 iou / giou 小于这个阈值就认为没有预测物体

# 学习率配置
lr_init = 2e-4                      # 初始学习率	# 0.00261
lr_lower = 1e-6                 # 最低学习率
lr_type = 'constant'   # 学习率类型 'exponential', 'piecewise', 'constant'
piecewise_boundaries = [10, 30]   # 单位:epoch, for piecewise
piecewise_values = [lr_init, 5e-4, 1e-5]

# 优化器配置
optimizer_type = 'momentum' # 优化器类型
momentum = 0.949          # 动量



# ############## 测试 ##############
score_thresh = 0.5      # 少于这个分数就忽略
iou_thresh = 0.5            # iou 大于这个值就认为是同一个物体
max_box = 50                # 物体最多个数
val_dir = "./test_pic"  # 测试文件夹, 里面存放测试图片
save_img = True             # 是否保存测试图片
save_dir = "./save"         # 图片保存路径
width = 416                     # 图片宽, 6G显存跑不起来 608 的, 哪位仁兄有更好的显卡可以试试
height = 416                    # 图片高


