# AEGI
Official implementation for the paper "***A Single-StreamArchitecture: AdversarialEncoder-Generator-Inference Networks***"

## Descriptions
This project is a [Pytorch](https://pytorch.org/) implementation of AEGI, which was published as a conference proceeding at CVPR 2021. This paper posessed a model that combines advantages of both VAEs and GANs. It maintains the training stability of VAEs and simultaneously demonstrates strong generative capability that allows for high-resolution image synthesis.
This code can run on single RTX2080ti for a short time to achieve the comparable visual effect as PGGAN or IntroVAE. e.g., 7 days for 1024×1024 celebA-HQ.

## How To Use This Code
You will need:
. [Pytorch](https://pytorch.org/), version 1.2.0
. torchvision, version 0.4.0
. numpy, opencv-python
