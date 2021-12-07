# SelfC
This is the PyTorch implementation of the the paper: Self-Conditioned Probabilistic Learning of Video Rescaling (ICCV 2021). [Paper pdf](https://arxiv.org/abs/2107.11639)

# Dependencies and Installation

- Python 3 (Recommend to use Anaconda)
- FFmpeg with H.265 codec activated

We recommend using the following compiling configuration to rigorously re-produce our results. We have verified that using different H.265 codec (for example, complied by the different configuration, different version ffmpeg, and different version gcc) would lead to slightly different results. The error will be exaggerated in high-bitrate settings.
Using the same H.265 codec in the testing as the training is important. 
```
ffmpeg version N-99732-g86267fc Copyright (c) 2000-2020 the FFmpeg developers
  built with gcc 7 (Ubuntu 7.5.0-3ubuntu1~18.04)
  configuration: --prefix=/home/tianyuan/ffmpeg_build --pkg-config-flags=--static --extra-cflags=-I/home/tianyuan/ffmpeg_build/include --extra-ldflags=-L/home/tianyuan/ffmpeg_build/lib --extra-libs='-lpthread -lm' --bindir=/home/tianyuan/bin --enable-gpl --enable-gnutls --enable-libass --enable-libfdk-aac --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 --enable-nonfree
  libavutil      56. 60.100 / 56. 60.100
  libavcodec     58.112.100 / 58.112.100
  libavformat    58. 63.100 / 58. 63.100
  libavdevice    58. 11.102 / 58. 11.102
  libavfilter     7. 88.100 /  7. 88.100
  libswscale      5.  8.100 /  5.  8.100
  libswresample   3.  8.100 /  3.  8.100
  libpostproc    55.  8.100 / 55.  8.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...
```

We also provide some evaluation log files in folder `test_logs`

- NVIDIA GPU + CUDA
We also provide the configuration file for quickly creating a conda env:
```
conda env create -f selfc_conda_config
```
# Data Preparation

Our framework is trained on the [Vimeo90K](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) dataset.

We also release the re-organized Vid4 and UVG datasets for facilitating future research.
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
## Compression network(Default mode)
You can test different bit-rates by changing the `h265_q` (crf parameter) in yml file.
```
cd codes
python test_compression.py -opt ./options/test/Selfc_H265/test_codec_uvg_bf.yml
```
The results should be the same as that in `test_logs/test_selfc_h265bf_q9.log`.
## Compression network(Only P-frame mode, a.k.a., Zero-latency + VeryFast Preset)
You can test different bit-rates by changing the `h265_q` (crf parameter) in yml file.
```
cd codes
python test_compression.py -opt ./options/test/Selfc_H265/test_codec_uvg_zerolatency.yml
```

# Train

## Rescaling network for SelfC-large model
```
cd codes
python -m torch.distributed.launch --nproc_per_node 2 --master_port 2478 train.py -opt ./options/train/train_rescaling_selfc_large.yml --launcher pytorch
```
## Compression network

We only train one compression model (Zero-latency, VeryFast Preset, crf =16) and test it under different bit-rate. Better results are expected to be achieved by training specified models for each bitrate.


```
cd codes
python -m torch.distributed.launch --nproc_per_node 2 --master_port 22637 train.py -opt ./options/train/train_compression.yml --launcher pytorch
```

# Known issues

Currently, the training of the compression model is rather slow, due to the slow online compression speed of the H.265 codec. Maybe fixed by using a faster implementation, such as the GPU accelerated H.265 codec.

# Citation
If you find our work useful, please star this repo and cite our paper. Thanks!
```
@inproceedings{tian2021self,
  title={Self-Conditioned Probabilistic Learning of Video Rescaling},
  author={Tian, Yuan and Lu, Guo and Min, Xiongkuo and Che, Zhaohui and Zhai, Guangtao and Guo, Guodong and Gao, Zhiyong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4490--4499},
  year={2021}
}

```