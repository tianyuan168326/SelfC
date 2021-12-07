# SelfC
This is the PyTorch implementation of paper: Self-Conditioned Probabilistic Learning of Video Rescaling (ICCV 2021). [Paper pdf](https://arxiv.org/abs/2107.11639)

# Dependencies and Installation

- Python 3 (Recommend to use Anaconda)
- FFmpeg with H.265 codec activated
- NVIDIA GPU + CUDA
We also provide the configuration file for quickly creating a conda env:
```
conda env create -f selfc_conda_config
```
# Data Preparation

Our framework is trained on the [Vimeo90K](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) dataset.

We also release the re-organized Vid4 and UVG datasets for facilitating the future research.
The two datasets are averagely divided into several groups without overlapping. Each group of Vid4 consists of 7 frames.
Each group of UVG consists of 100 frames, which is consistent with the previous learnable video compression methods such as DVC and FVC.
## Download link
[Reorg Vid4](https://pan.baidu.com/s/1sjEDtPgyfZjZRZTX8TF9hA) Password: 7iqd


[Reorg UVG](https://pan.baidu.com/s/13EdBLPmP4yTciL0FlhHEFQ) Password: uic6

# Test

Firstly, we download the pretrained models and save them to the folder `pretrained_models`.

[Video rescaling model](https://pan.baidu.com/s/1XUmSDDzjaxfNxS4iS9se3w)(Password: e1xe)
[Video compression model](https://pan.baidu.com/s/1M2rkLIW1dt1xWEywJwne5A)(Password: j5ff)

## Rescaling network for SelfC-large model
```
cd codes
python test_rescaling.py -opt ./options/test_v2/rescaling/test_SelfC_large_vid4.yml
```
## Compression network(B-frame mode)
```
cd codes
python test_compression.py -opt ./options/test/Selfc_H265/test_codec_uvg_bf.yml
```

## Compression network(Only P-frame mode, a.k.a., Zero-latency)
```
cd codes
python test_compression.py -opt ./options/test/Selfc_H265/test_codec_uvg_zerolatency.yml
```

# Train

## Rescaling network for SelfC-large model
```
cd codes
python train.py -opt ./options/train/train_SelfC_large_GMM_STP6.yml
```
## Compression network

We only train one compression model and test it under different bit-rate. Better results are expected to be achieved by training specified model for each bit-rate.


```
cd codes
python test_compression.py -opt ./options/train/train_compression.yml
```

# Known issues

Currently, the training of the compression model is rather slow, due to the slow online compression speed of the H.265 codec. Maybe fixed by using a faster implementation, such as the GPU accelerated H,265 codec.

# Citation
If you find our work usefull, please star this repo and cite our paper. Thanks!
```
@inproceedings{tian2021self,
  title={Self-Conditioned Probabilistic Learning of Video Rescaling},
  author={Tian, Yuan and Lu, Guo and Min, Xiongkuo and Che, Zhaohui and Zhai, Guangtao and Guo, Guodong and Gao, Zhiyong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4490--4499},
  year={2021}
}

```