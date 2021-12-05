import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os
import time
from global_var import GlobalVar
def h265_xxx( input):
    input = input.clone().detach()
    dev_id = str(input.device)
    input = torch.clamp(input, 0, 1)
    # output = (input * 255.).round() 
    output = input*255
    # output = input*255.0
    bt,c,h,w =output.size()
    # t = 7
    # b = bt//7 
    # print("output",output.size()) ## torch.Size([7, 3, 144, 176])
    # exit()
    frames = output.permute(0,2,3,1)
    frames = frames.cpu().numpy().astype(np.uint8)
    # video_name = "outputvideo.mp4"
    video_name = "./tmp/outputvideo"+dev_id+Quantization_H265_Suggrogate.file_random_name+"cvb.h265"
    # video_name = "./tmp/cvb.h265"
    



    p = {
        '-c:v': 'libx265',
        '-preset':'veryfast',
        '-tune':'zerolatency',
        "-s":str(w)+"x"+str(h),
    '-pix_fmt': 'yuv420p',
    # '-threads':"1",
    "-vframes":"10000",
    # "-profile:v": "main",
    # "-level:v":"4.0",
    # "-r":"50",
    #  "-qp":"0",
    #  "-x265-params":"crf=0:lossless=1:preset=veryfast:qp=0"
        "-x265-params":"crf="+str(Quantization_H265_Suggrogate.q)+\
            ":keyint="+str(Quantization_H265_Suggrogate.keyint)+":no-info=1"
    }
    import time
    T1 = time.time()
    # time.sleep(10000000)
    # p = {
    #  '-c:v': 'libx264',  '-pix_fmt': 'yuv444p',
    #  "-crf":"0"
    # }
    writer = skvideo.io.FFmpegWriter(video_name,outputdict = p,verbosity = 0)
    for i in range(bt):
        writer.writeFrame(frames[i, :, :, :])
    writer.close()
    file_size = os.path.getsize(video_name)
    bpp = file_size*8.0/(h*w*Quantization_H265_Suggrogate.scale_times*Quantization_H265_Suggrogate.scale_times*GlobalVar.get_Temporal_LEN())
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(video_name,
                    inputdict= {},
                    outputdict={})
    # iterate through the frames
    decoded_frames = []
    for frame in reader.nextFrame():
        # do something with the ndarray frame
        # decoded_frames += [torch.from_numpy(frame).cuda(input.device)]
        decoded_frames += [torch.from_numpy(frame)]
    # print('runing time2 %s ms' % ((T3 - T2)*1000))
    decoded_frames = torch.stack(decoded_frames,dim=0).cuda(input.device)
    decoded_frames = decoded_frames.permute(0,3,1,2)
    decoded_frames = decoded_frames/255.
    # return output/255.
    return decoded_frames
   
import random
from models.modules.Subnet_constructor import *
class Quantization_H265_Suggrogate(nn.Module):
    def __init__(self,q = 17,keyint = 12,scale_times = 2):
        super(Quantization_H265_Suggrogate, self).__init__()
        Quantization_H265_Suggrogate.q = q
        Quantization_H265_Suggrogate.keyint = keyint
        Quantization_H265_Suggrogate.scale_times = scale_times
        Quantization_H265_Suggrogate.file_random_name = None
        mid_c = 24
        self.suggrogate_net = nn.Sequential(
            DenseBlock(3,mid_c,INN_init=False),
            DenseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=False),
            DenseBlock(mid_c,mid_c,INN_init=False),
            DenseBlock(mid_c,3,INN_init=False),
        )
    def forward(self, input):
        if not Quantization_H265_Suggrogate.file_random_name:
            Quantization_H265_Suggrogate.file_random_name = str(time.time())
        H265_encoder_encoder_out = h265_xxx(input).detach()
        sug_out = self.suggrogate_net(input)
        mimick_loss = torch.mean((H265_encoder_encoder_out - sug_out)**2.0)
        return sug_out,mimick_loss
