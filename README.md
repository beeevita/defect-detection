## 工业缺陷检测

图像分类任务

### 1.文件介绍

#### 1.1 main.py

自行实现的主程序，内有所有改进模型的训练，以及ROC曲线绘制和AUC值的计算。

#### 1.2 res2net_3_0.py

res2net、以及Bi-res2net和其他改进的模型定义，可以选择`model_type`中的几种模型运行。

#### 1.3 resnet.py

resnet模型定义

#### 1.4 print.py

对输出的存有训练准确率及loss的txt文件进行可视化展示

#### 1.5 draw_feature.py

对训练好的模型进行特征图片展示，时间原因，在汇报中未展示，但是代码可运行

### 2.文件夹

#### 2.1 feautures

存储各模型特征图

#### 2.2 result

acc loss 训练过程可视化

#### 2.3 weights

训练好的模型参数



