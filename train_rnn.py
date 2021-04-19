#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/19/21-13:19
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : train_rnn.py
# @Project  : 00PythonProjects
import argparse
import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from model import TextRNN

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--BASE_DIR', help='inner batch size', default="cnews", type=str)
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
parser.add_argument('--MODEL', help='inner batch size', default='checkpoints/textrnn', type=str)
parser.add_argument('--TENSORBOARD', help='inner batch size', default='tensorboard/textrnn', type=str)

args = parser.parse_args()
print(args.__dict__)

# base_dir = 'cnews'
train_dir = os.path.join(args.BASE_DIR, 'cnews.train.txt')
test_dir = os.path.join(args.BASE_DIR, 'cnews.test.txt')
val_dir = os.path.join(args.BASE_DIR, 'cnews.val.txt')
vocab_dir = os.path.join(args.BASE_DIR, 'cnews.vocab.txt')

save_path = os.path.join(args.MODEL, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    # tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(args.TENSORBOARD):
        os.makedirs(args.TENSORBOARD)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.TENSORBOARD)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(args.MODEL):
        os.makedirs(args.MODEL)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, args.SEQ_LENGTH)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, args.SEQ_LENGTH)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(args.NUM_EPOCHS):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, args.BATCH_SIZE)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, args.DROPOUT_KEEP_PROB)

            if total_batch % args.SAVE_PER_BATCH == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % args.PRINT_PER_BATCH == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = args.DROPOUT_KEEP_PROB
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, args.SEQ_LENGTH)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':

    print('Configuring RNN model...')
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, args.VOCAB_SIZE)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    args.VOCAB_SIZE = len(words)
    model = TextRNN(args)

    if args.DO_TRAIN:
        train()
    if args.DO_TEST:
        test()
