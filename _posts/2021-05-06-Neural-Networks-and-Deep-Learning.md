---
title: Deep Learning
layout: post
category: Note
---

# Logistic Regression as a Neural Network

## Basic of Neural Network Programming

## Python and Vectorization

-SIMD(Single Instruction Multiple Data)

- 使用numpy而不是显式for loop来提高速度
- Broadcast
- 加入reshape
- Logistic Regression Cost的合理性（一个角度是最大似然估计，MLE）

### Programming Tips

- 不要使用rank 1 array，使用reshape
- 使用assert

## Gradient Desent for Neural Networks

- keepdims=True防止python出现rank 1 array，另外一个选择是reshape
- 使用计算图计算导数
- 使用相对小的数值初始化参数，防止进入饱和区

### little question

为什么neural network不能将参数全部初始化为0而logistic可以（因为RELU？）\
ok，是因为对称性的问题，如果全部初始化为0，hidden layer的所有神经元参数就全部都一样了

## Deep L-layer Neural Network

### Debug Tip

关注matrix的dimension

## little summary

&emsp;&emsp;主要就是了解了整个deep neural network的工作流程。forward prop、backward prop，在前向传播过程中cache一些中间变量，在反向传播中利用以更方便的计算梯度。正向传播并没有什么问题，反向传播可以看作一个个building block堆砌（loop）起来。具体来说每一个block以上一层的$dA$和对应前向传播layer的cache作为输入，计算得到$dZ^[l],dW^[l],db^[l],dA[l-1]$，以此类推，就可以得到所有的梯度。

# Improving Deep Neural Network

## Setting up your Machine Learning Application

### Bias and Variance

bias取决于模型在训练集上的准确率高低，variance取决于模型在训练集和验证集上的准确率差异。根据不同的情况，选择不同的应对策略。例如，bias大，使用更大的网络，variance大，使用正则化方法。

### Regularization

- Frobenius norm: 所有W矩阵的所有元素平方相加\
- inverted dropout：在前向传播过程，使用mask关闭一些neuron，再将a除以概率；在反向传播过程中，也使用相同的mask关闭dA，并同样使用dA除以概率。（QUESTION）在test时，不做任何处理。同时，对于担心overfitting的地方可以使用小的keep-prob，对于不太担心overfitting的地方可以使用大的keep-prob。同时，由于dropout可能会使$J$不再单调，所以一个比较好的做法是首先不使用dropout来确保训练过程确实正确（$J$在不断减小），之后再使用dropout缓和overfitting（如果确实存在）。
- early stop：当验证集上的loss开始上升时，及时停止train
与norm相比，early stop将优化$J$和选择最优超参数混合在一起，而norm则是将两个过程分开\

### Setting up your Optimization Problem

- Normalize:减去均值，除以方差根。**使用相同的$\mu, \sigma$normalize测试集**，使得loss function更加容易优化（W和b都在大致相同的区间）\
- Weight Initialization，为了**缓解**梯度消失和梯度爆炸，可以使用一些参数初始化方法，例如若使用RELU作为激活函数，则可以使用方差为$\frac{2}{n^[l-1]}$（He Initialization）的参数；如果使用tanh作为激活函数，则可以使用方差为$\frac{1}{n^[l-1]}$的参数（Xavier Initialization）。
- Gradient Checking：

### Hyperparameters and Optimization Algorithms

- mini-batch gradient descent。epoch对应遍历一遍数据集，例如mb gd将数据集分为$T$ 个mini-batch，那么一个epoch会将参数更新$T$次。特别的，当mini-batch的size为$m$时，算法就是batch gradient descent；如果mini-batch的size为1时，算法为Stochastic gradient descent（）。bgd每次迭代花费的时间太长，sgd失去了vectorization的优势，所以实践中将size选为中间的某个值。
- Exponentially Weighted (Moving) Averages。$V_{t} = \beta\times v_{t-1} + (1 - \beta)\times \theta_{t}$，大致表示$t$到$t - \frac{1}{1-\beta}$这段时间内的平均值。实际上可以看成数据点向量和指数递减向量的内积。
- Bias Correction。由于初始化$v_{0}=0$，所以在最开始的阶段$v_{t}$都十分小，所以需要修正。修正方法就是使用$\frac{v_{t}}{1-\beta^{t}}$来代替$v_{t}$。合理性在于当$t$比较小时（在初始阶段），分母比较接近0，会使得到的值变大，当$t$较大时，分母比较接近1，而不会对得到的值产生太大影响。但是在实际应用中，人们使用这个correction并不多...
- momentum。实际上就是计算了梯度的Exponentially Weighted (Moving) Averages。作用在于不同的参数也许需要不同的learning rate，但是在实际操作中一般只会设计一个learning rate，所以对一些梯度较大的参数，这个学习率可能过大了，因而出现**振荡**。而通过计算EWA，也就是一些mini-batch的梯度平均值，可以缓解这种振荡（正负相抵消），从而加快了收敛过程。
- RMSprop（root mean square prop）。实际上就是计算了梯度的平方的Exponentially Weighted (Moving) Averages。之后在更新参数时使用$\frac{dW}{\sqrt{S_{dW} + \epsilon}}$来代替$dW$。也就是将较大的梯度除以较大的数来防止"走的过多"，将较小的梯度除以较小的数来加快收敛。
- Adam（Adaptive Moment Estimation）。结合了momentum和RMSprop（都结合了bias correction）。
- Learning rate decay。

## Hyperparameter Tuning, Batch Normalization and Programming Frameworks

### Hyperparameter Tuning

- 因为通常有很多超参数要找，所以grid search通常计算量过大，所以通常使用**random sample**或者**先粗后细**的方法来寻找最好的超参数。
- 有一些超参数可以直接通过均匀分布采样得到样本（比如网络的层数、每层的宽度等），但是有一些不可以（比如学习率）。对于类似学习率的超参数，通过对指数进行采样，从而得到对各个数量级均有考虑的采样结果。

### Batch Normalization

- 基本做法就是对于每一层产生的$Z^{[l]}$，对于每一个分量减去均值、除以方差，在赋予一个新的均值和方差（通过element-wise乘法和加法），得到$\tilde{Z}^{[l]}$，再继续进行后续的步骤。特别的，使用BN时可以省略参数$b$。
- 在test中，使用在训练中计算好的每一层的$\mu^{[l]}, \sigma^{[l]}$，使用这个值来做normalization。

### Multi-class Classification

-


------------
### Mismatched Training and Dev/Test Set
- Human level(avoidable bias)Trian set(variance)Train-dev set(data mismatched)dev set(degree of overfitting to dev set)test set 
- Addressing data mismatch。
### Learning from Multiple Tasks
- 实际上就是将最后一层改为softmax（与之相对的是hard max，直接产生one hot向量）。具体来说就是首先通过矩阵乘法产生$Z^{[l]}$，之后在通过公式$t = e^{Z^{[l]}}$以及一个正则化的操作最终产生$A^{[l]}$。

## Structuring Machine Learning Projects

### Introduction 2 ML Strategy

- Orthogonalization。将各个目标分离开来，在调节一个目标的同时而不会影响另外一个目标。

### Single Number Evaluation Metric

- 多个指标一般难以比较，所以使用单一的指标。例如对多个指标取平均值，或者计算多个指标的$F1$ score。
- Satisficing and Optimizing Metric。当有多个需要考虑的指标时，将最重要的设置为Optimizing Metric（最优化），将其他的设置为Satisficing Metric（满足一定条件即可）。这样更方便考虑。
- Train/Dev/Test Distrbutions。Dev set和Test set应该来自相同的Distribution。
- ***first place target, then shot at target***（以终为始哈哈哈哈哈哈哈哈）

### Comparing 2 Human-level Performance
- Bayes error。理论上最小的误差。可以使用Human-level来近似。
- 对于相同的training error和dev error，不同的human performance，侧重点（avoidable bias和variance）会不同。
- 两个监督学习的基本假设
    - 你可以在训练集上拟合得足够好
    - 训练集上的性能可以很好得泛化到验证、测试集上
- 从而对应了两个优化的方向——减小available bias和减小variance。

### Error Analysis

- 通过查看测试集上结果错误的样例，分析得到算法可以改进的方向以及对应的潜在提升空间。
- Deep Learning算法对于数据集的random error鲁棒性较强，但是对于systemic error鲁棒性较弱。
- 构建一个Deep Learning系统的流程（Build your first system quickly, then iterate）：
    1. 构造数据集
    2. 迅速搭建初始系统
    3. 使用Bias/Variance analysis和Error analysis来决定后续步骤
    4. 迭代
