# SelfC
This is the PyTorch implementation of paper: SelfC

# Data Preparation
Our framework is trained on the [Vimeo90K](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) dataset.

We also release the re-organized Vid4 and UVG dataset for facilitate the future research.
The two dataset are averaged divided into several groups without overlapping. Each group of Vid4 consists of 7 frames.
Each group of UVG consists of 100 frames, which is consistent with the previous learnable video compression methods such as DVC and FVC.
## Download link
Vid4: Link: https://pan.baidu.com/s/1sjEDtPgyfZjZRZTX8TF9hA 
Password: 7iqd

UVG: 

# Test

Firstly, we download the pretrained models and save them to the folder `pretrained_models`.

[Link for video rescaling model](https://pan.baidu.com/s/1XUmSDDzjaxfNxS4iS9se3w)(Password: e1xe)
[Link for video compression model](https://pan.baidu.com/s/1M2rkLIW1dt1xWEywJwne5A)(Password: j5ff)
## Rescaling network for SelfC-large model
```
python test_rescaling.py -opt ./options/test_v2/rescaling/test_SelfC_large_vid4.yml
```
## Compression network
```
python test_rescaling.py -opt ./options/test_v2/rescaling/test_SelfC_large_vid4.yml
```