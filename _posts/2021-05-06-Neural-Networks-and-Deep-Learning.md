# Logistic Regression as a Neural Network

## Basic of Neural Network Programming
## Python and Vectorization
SIMD(Single Instruction Multiple Data)
使用numpy而不是显式for loop来提高速度
Broadcast
加入reshape
Logistic Regression Cost的合理性（一个角度是最大似然估计，MLE）
### Programming Tips:
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
