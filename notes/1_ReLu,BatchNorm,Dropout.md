# ReLu, BatchNorm与Dropout

## 1. 基本内容

1. Batch Normalization（BN）即批规范化，是正则化的一个重要手段。在正则化效果的基础上，批处理规范化还可以减少卷积网络在训练过程中的梯度弥散。这样可以减少训练时间，提升结果。第一步的规范化会将几乎所有数据映射到激活函数的非饱和区（线性区），仅利用到了线性变化能力，从而降低了神经网络的表达能力。而进行再变换，则可以将数据从线性区变换到非线性区，恢复模型的表达能力。
2. Dropout是神经网络中防止模型过拟合的重要正则化方式，通过随机废弃神经元防止过拟合，一般情况下，只需要在网络存在过拟合风险时才需要实现正则化。如果网络太大、训练时间太长、或者没有足够的数据，就会发生这种情况。注意，*这只适用于CNN的全连接层。对于所有其他层，不应该使用Dropout*。相反，应该在卷积之间插入批处理标准化。这将使模型规范化，并使模型在训练期间更加稳定。
3. ReLU是激活函数的一种。

## 2. 适用范围

### 2.1 BatchNorm不适用的情况

1. BN并不是适用于所有任务的，在image-to-image这样的任务中，尤其是超分辨率上，图像的绝对差异显得尤为重要，所以batchnorm的scale并不适合
2. BN会引入噪声（因为是mini batch而不是整个training set），所以对于噪声敏感的方法（如RL）

### 2.2 BatchNorm和Dropout冲突

1. BN或Dropout单独使用能加速训练速度并且避免过拟合,但是倘若一起使用，会产生负面效果。BN在某些情况下会削弱Dropout的效果
2. BN与Dropout最好不要一起用，若一定要一起用，有[两种方法][1]:(a)在所有BN层后使用Dropout;(b)修改Dropout公式（如使用高斯Dropout）使得它对方差不是那么敏感。总体思路是降低方差偏移

[1](https://zhuanlan.zhihu.com/p/33101420)
[2](https://zhuanlan.zhihu.com/p/33173246)
[3](https://zhuanlan.zhihu.com/p/61725100)
[4](https://zhuanlan.zhihu.com/p/32230623)
[5](https://blog.csdn.net/clearch/article/details/80266622)