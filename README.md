# AEGI
Official implementation for the paper "***A Single-StreamArchitecture: AdversarialEncoder-Generator-Inference Networks***"

## Descriptions
This project is a [Pytorch](https://pytorch.org/) implementation of AEGI, which was published as a conference proceeding at CVPR 2021. This paper posessed a model that combines advantages of both VAEs and GANs. It maintains the training stability of VAEs and simultaneously demonstrates strong generative capability that allows for high-resolution image synthesis.

This code can run on single RTX2080ti for a short time to achieve the comparable visual effect as PGGAN or IntroVAE. e.g. 7 days for 1024×1024 celebA-HQ.

## How To Use This Code
You will need:
  - Python 3.7
  - [Pytorch](https://pytorch.org/), version 1.2.0
  - torchvision, version 0.4.0
  - numpy, opencv-python

The default parameters for CelebA-HQ faces at 128x128, 256x256, 512x512 and 1024x1024 resolutions are provided in the file 'run_128.sh', 'run_256.sh', 'run_512.sh' and 'run_1024.sh', respectively. 

 To train 128x128 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=6 python3 main.py --dataroot=your_data_path  --noise_dim=256  --batch_size=128  --test_batch_size=32  --nEpochs=500  --save_step=2  --channels='32, 64, 128, 256, 512'  --trainsize=29000  --input_height=128  --output_height=128  --m_plus=140  --weight_neg=0.5  --weight_rec=0.2  --weight_kl=1.  --num_vae=10  --num_gan=10 > main.log 2>&1 &
