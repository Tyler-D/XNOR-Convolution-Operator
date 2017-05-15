# XNOR-Convolution-Operator
raw implementation of XNOR convolution's operators including weights and input's binarization, im2col and binary convolution.

The idea is from [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279).
However, it seems like that we cannnot handle the padding convolution with only xnor and popcnt operation of binary codes.
