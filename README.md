### CNN&RNN&LSTM&GRU实现对短文本的分类任务

===========================
#### 00-项目信息
```
作者：TuringEmmy
时间:2021-04-19 15:01:27
详情： 针对短文本分类，分别采用卷积神经网络，循环神经网络对其进行embedding,只有对其进行分类任务
```
#### 01-环境依赖
```
ubuntu18.04
python3.7
tensorflow1.14
```
#### 02-部署步骤
##### 脚本
```
sh scripts/train_cnn.sh  
sh scripts/train_rnn.sh
sh scripts/predict_cnn.sh  
sh scripts/predict_rnn.sh  
sh scripts/tensorboard_cnn.sh  
sh scripts/tensorboard_rnn.sh  
sh scripts/api_cnn.sh  
sh scripts/api_rnn.sh  

```

#### 03-目录结构描述
```
.
├── cnews_loader.py
├── model.py
├── predict_cnn.py
├── predict_rnn.py
├── README.md
├── scripts
│   ├── api_cnn.sh
│   ├── api_rnn.sh
│   ├── predict_cnn.sh
│   ├── predict_rnn.sh
│   ├── tensorboard_cnn.sh
│   ├── tensorboard_rnn.sh
│   ├── train_cnn.sh
│   └── train_rnn.sh
├── server_cnn.py
├── server_rnn.py
├── train_cnn.py
└── train_rnn.py

```


#### 04-版本更新
##### V1.0.0 版本内容更新-2021-04-19 15:05:59
- 对模型的训练测试完毕
- 加载模型，对新数据进行预测
##### V1.0.1 版本内容更新-2021-04-19 15:36:25
- api形式接口实现()
- 根据项目需要，后期在docker

#### 05-TUDO
- 以api的形式呈现
- 部署到docker内部