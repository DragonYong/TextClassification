#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/19/21-15:07
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : server_cnn.py
# @Project  : 00PythonProjects
import argparse
import json
import os

from flask import Flask, request

from predict_cnn import CnnModel

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--BASE_DIR', help='inner batch size', default="cnews", type=str)
parser.add_argument('--EMBEDDING_DIM', help='词向量维度', default=64, type=int)
parser.add_argument('--SEQ_LENGTH', help='序列长度', default=600, type=int)
parser.add_argument('--NUM_CLASSES', help='类别数', default=10, type=int)
parser.add_argument('--NUM_FILTERS', help='滤波器书', default=256, type=int)
parser.add_argument('--KERNEL_SIZE', help='滤波器大小', default=5, type=int)
parser.add_argument('--VOCAB_SIZE', help='词汇表达小', default=5000, type=int)
parser.add_argument('--NUM_LAYERS', help='隐藏层层数', default=2, type=int)
parser.add_argument('--HIDDEN_DIM', help='隐藏层神经元', default=128, type=int)
parser.add_argument('--DROPOUT_KEEP_PROB', help='dropout保留比例', default=0.8, type=float)
parser.add_argument('--LEARNING_RATE', help='学习率', default=1e-3, type=float)
parser.add_argument('--BATCH_SIZE', help='每批训练大小', default=128, type=int)
parser.add_argument('--NUM_EPOCHS', help='总迭代轮次', default=10, type=int)
parser.add_argument('--PRINT_PER_BATCH', help='每多少轮输出一次结果', default=100, type=int)
parser.add_argument('--SAVE_PER_BATCH', help='每多少轮存入tensorboard', default=10, type=int)
parser.add_argument('--MODEL', help='每多少轮存入tensorboard', default='checkpoints/textcnn', type=str)
parser.add_argument('--DO_TRAIN', help='训练', action='store_true')
parser.add_argument('--DO_TEST', help='测试', action='store_true')

args = parser.parse_args()

@app.route('/do', methods=['POST'])
def event_extraction():
    data = request.data
    print(data, type(data))
    print(eval(data)["text"])
    cnn_model = CnnModel()
    cnn_model.config = args
    event = {"text": eval(data)["text"], "result": cnn_model.predict(eval(data)["text"])}
    return json.dumps(event, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
