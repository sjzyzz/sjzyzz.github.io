---
title: Paper Reading
layout: post
category: academic
---
### More Is Less: Learning Efficient Video Representations by Big-Little Network and Depthwise Temporal Aggregation

secret weapon: 一个新的网络架构（big-little-video-net），一个分支（更加深）处理低分辨率的序列，一个分支（更加紧凑）处理高分辨率的序列。以及一个合并时间信息的模块——temporal aggregation module，通过$1\times1$depthwise卷积来实现时间信息的融合。从而避免了使用3D卷积，降低了模型复杂度，同时允许视频的更多帧作为输入。

does it work? 根据论文的实验结果，在TSN的基础上加入TAM模块其实可以增强模型的表现。以及以TSN为框架，bLVNet的表现其实强于resnet的表现。

why it works? 具体的网络框架为何work，可能需要查看另外的一篇论文。对于TAM，可以看做实在TSM的基础上进行的扩展，<span style="color:blue">至于具体的联系，还需要再仔细思考</span>。

### Temporal Relational Reasoning in Videos

secret weapon: 实际上提出了一种新的特征fusion的方法，从而达到了增强temporal编码的作用。具体来说首先计算多个帧数阶的relation（例如2-frame relation， 3-frame relation，对于每一种relation计算多组），在一个帧对中，帧与帧之间的relation通过MLP进行计算，不同帧对再次通过MLP得到最终的对应帧数阶relation。最后再将各个阶的relation相加，得到最终结果。<span style="color:purple">在计算帧对的relation时可以考虑借鉴SDN的想法，使用差值</span>。

does it work? 首先，如何才能证明这个东西work？fine，他们是提出了一个新的架构，并不是一个即插即用的模块，所以对比方法只能是和以前的sota作比较。其实论文里的实验部分偏少，也没有做ablation analysis...只是与TSN做了一个对比。

why it works? 

### Gate-Shift Networks for Video Action Recognition

<span style="color:purple">视频的采样策略也许是一个增长点</span>

<span style="color:purple">决定哪些过去和将来的帧中的信息是对现在的帧中有用的</span>

secret weapon: Gate-Shift Module，首先通过$3\times3\times3$的卷积得到一个gate以判断哪些特征是需要被shift的，之后将需要shift的部分进行shift，再加上原来不需shift的部分，就得到了输出。

does it works? 根据paper上的实验数据，将GSM加到TSN上，使得模型的temporal model能力大为增强。所以确实可以说，GSM是work的。<span style="color: yellow">但是实际上它的创新点在于gate，那么gate是否起作用呢？paper中并没有比较相同情况下GSM与TSM的性能比较。</span>

why it works? 我认为gate的出发点是好的，不是人为的设置一个比例进行shift（TSM的方式），而是生成一个gate，将一部分特征进行shift，一部分特征保持不变。

### X3D: Expanding Architectures for Efficient Video Recognition

### STM: SpatioTemporal and Motion Encoding for Action Recognition

### Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks