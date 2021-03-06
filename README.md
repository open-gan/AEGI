# AEGI
Official implementation for the paper "***A Single-StreamArchitecture: AdversarialEncoder-Generator-Inference Networks***"

## Descriptions
This project is a [Pytorch](https://pytorch.org/) implementation of AEGI. This paper posessed a model that combines advantages of both VAEs and GANs. It maintains the training stability of VAEs and simultaneously demonstrates strong generative capability that allows for high-resolution image synthesis.

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
CUDA_VISIBLE_DEVICES=0 python3 main.py  --dataroot='/dataset/celebAHQ/celeba-128/'  --noise_dim=256  --batch_size=128  --test_batch_size=32  --nEpochs=500  --save_step=2  --channels='32, 64, 128, 256, 512'  --trainsize=29000  --input_height=128  --output_height=128  --m_plus=140  --weight_neg=0.5  --weight_rec=0.2  --weight_kl=1.  --num_vae=10  --num_gan=10  --weight_EM=0.999 > main.log 2>&1 &
```

 To train 256x256 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py  --dataroot='/dataset/celebAHQ/celeba-256/'  --noise_dim=512  --batch_size=32  --test_batch_size=16  --nEpochs=500  --save_step=2  --channels='32, 64, 128, 256, 512, 512'  --trainsize=29000  --input_height=256  --output_height=256  --m_plus=300  --weight_neg=0.5  --weight_rec=0.1  --weight_kl=1.  --num_vae=10  --num_gan=10 --weight_EM=0.999 > main.log 2>&1 &
```

 To train 512x512 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py  --dataroot='/dataset/celebAHQ/celeba-512/'  --noise_dim=512  --batch_size=16  --test_batch_size=16  --nEpochs=500  --save_step=2  --channels='16, 32, 64, 128, 256, 512, 512'  --trainsize=29000  --input_height=512  --output_height=512  --m_plus=100  --weight_neg=0.25  --weight_rec=0.01  --weight_kl=1.  --num_vae=0  --num_gan=60  --weight_EM=0.999 > main.log 2>&1 &
```

 To train 1024x1024 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py  --dataroot='/dataset/celebAHQ/celeba-1024/'  --noise_dim=512  --batch_size=5  --test_batch_size=5  --nEpochs=500  --save_step=2  --channels='16, 32, 64, 128, 256, 512, 512, 512'  --trainsize=29000  --input_height=1024  --output_height=1024  --m_plus=160  --weight_neg=0.5  --weight_rec=0.0025  --weight_kl=1.  --num_vae=2  --num_gan=4  --weight_EM=0.999 > main.log 2>&1 &
```

## Results
[![logo](https://github.com/open-gan/AEGI/blob/main/Samples/AEGI_1.png)](https://github.com/open-gan/AEGI/blob/main/Samples/AEGI_1.png) 

## Citation
If you find our code helpful in your research or work please cite our paper.
```
@inproceedings{AEGI,
  title={A Single-StreamArchitecture: AdversarialEncoder-Generator-Inference Networks},
  author={*},
  booktitle={*},
  pages={*},    
  year={2021}
}
```

**The released codes are only allowed for non-commercial use.**
