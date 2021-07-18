---
title: Paper Reading
layout: post
category: academic
---
### More Is Less: Learning Efficient Video Representations by Big-Little Network and Depthwise Temporal Aggregation

secret weapon: 一个新的网络架构（big-little-video-net），一个分支（更加深）处理低分辨率的序列，一个分支（更加紧凑）处理高分辨率的序列。以及一个合并时间信息的模块——temporal aggregation module，通过$1\times1$depthwise卷积来实现时间信息的融合。从而避免了使用3D卷积，降低了模型复杂度，同时允许视频的更多帧作为输入。

does it work? 根据论文的实验结果，在TSN的基础上加入TAM模块其实可以增强模型的表现。以及以TSN为框架，bLVNet的表现其实强于resnet的表现。

why it works? 具体的网络框架为何work，可能需要查看另外的一篇论文。对于TAM，可以看做实在TSM的基础上进行的扩展，***至于具体的联系，还需要再仔细思考***。

### Temporal Relational Reasoning in Videos

secret weapon: 