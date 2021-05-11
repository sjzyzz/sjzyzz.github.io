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
