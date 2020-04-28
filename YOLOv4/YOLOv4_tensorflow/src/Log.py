# coding:utf-8
# 日志记录
import config
from os import path
import os
from utils import tools

# 添加一条日志
def add_log(content):
    if not path.isdir(config.log_dir):
        os.mkdir(config.log_dir)
        add_log("message:创建'{}'文件夹".format(config.log_dir))
    log_file = path.join(config.log_dir, config.log_name)
    print(content)
    tools.write_file(log_file, content, True)
    return

# 添加一条损失
def add_loss(value):
    if not path.isdir(config.log_dir):
        os.mkdir(config.log_dir)
        add_log("创建'{}'文件夹".format(config.log_dir))
    loss_file = path.join(config.log_dir, config.loss_name)
    tools.write_file(loss_file, value, False)
    return