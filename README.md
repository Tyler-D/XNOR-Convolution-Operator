# XNOR-Convolution-Operator
raw implementation of XNOR convolution's operators including weights and input's binarization, im2col and binary convolution.

The idea is from [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279).
However, it seems like that we cannnot handle the padding convolution with only xnor and popcnt operation of binary codes.

## speed test
comparing with caffe(make with openblas, -o2).

Enviroment |
----------|
Ubuntu 14.04| 
Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz|

Test: input 64x128x128; kernel 1x3x3; stride 1; padding 0

platform|speed/ms
------|-------
caffe| 20|
raw test| 43
caffe with this xnor-conv | 15
(I merge the operator into caffe and it's faster than raw test, emmmmm.......)
