#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/19/21-13:09
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : predict_rnn.py
# @Project  : 00PythonProjects
# coding: utf-8

from __future__ import print_function

import argparse
import os

import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnews_loader import read_category, read_vocab
from model import TextRNN

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--BASE_DIR', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--EMBEDDING_DIM', help='词向量维度', default=64, type=int)
parser.add_argument('--SEQ_LENGTH', help='序列长度', default=600, type=int)
parser.add_argument('--NUM_CLASSES', help='类别数', default=10, type=int)
parser.add_argument('--VOCAB_SIZE', help='词汇表达小', default=5000, type=int)
parser.add_argument('--NUM_LAYERS', help='隐藏层层数', default=2, type=int)
parser.add_argument('--HIDDEN_DIM', help='隐藏层神经元', default=128, type=int)
parser.add_argument('--RNN', help='LSTM or GRU ', default="LSTM", type=str)
parser.add_argument('--DROPOUT_KEEP_PROB', help='dropout保留比例', default=0.8, type=float)
parser.add_argument('--LEARNING_RATE', help='学习率', default=1e-3, type=float)
parser.add_argument('--BATCH_SIZE', help='每批训练大小', default=128, type=int)
parser.add_argument('--NUM_EPOCHS', help='总迭代轮次', default=10, type=int)
parser.add_argument('--PRINT_PER_BATCH', help='每多少轮输出一次结果', default=100, type=int)
parser.add_argument('--SAVE_PER_BATCH', help='每多少轮存入tensorboard', default=10, type=int)
parser.add_argument('--DO_TRAIN', help='训练', action='store_true')
parser.add_argument('--DO_TEST', help='测试', action='store_true')
parser.add_argument('--MODEL', help='LSTM or GRU ', default='checkpoints/textrnn', type=str)

args = parser.parse_args()

vocab_dir = os.path.join(args.BASE_DIR, 'cnews.vocab.txt')

save_path = os.path.join(args.MODEL, 'best_validation')  # 最佳验证结果保存路径


class RnnModel:
    def __init__(self):
        self.config = args
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        data = [self.word_to_id[x] for x in message if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.SEQ_LENGTH),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    rnn_model = RnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(rnn_model.predict(i))
