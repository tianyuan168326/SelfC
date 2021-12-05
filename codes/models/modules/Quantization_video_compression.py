import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os
from global_var import GlobalVar
import random
import time
class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        dev_id = str(input.device)
        input = torch.clamp(input, 0, 1)
        output = input*255
        bt,c,h,w =output.size()
        frames = output.permute(0,2,3,1)
        frames = frames.cpu().numpy().astype(np.uint8)
        video_name = "./tmp/outputvideo"+dev_id+Quantization_H265.file_random_name+"cvb.h265"
        
   
        if Quantization_H265.q == "dynamic":
            local_q = random.randint(8,35)
        else:
            local_q = Quantization_H265.q
        p = {
         '-c:v': 'libx265',
         '-preset':'veryfast',
         '-tune':'zerolatency',
         "-s":str(w)+"x"+str(h),
        '-pix_fmt': 'yuv444p',
        # '-threads':"1",
        "-vframes":"100",
        # "-profile:v": "main",
        # "-level:v":"4.0",
        # "-r":"50",
        #  "-qp":"0",
        #  "-x265-params":"crf=0:lossless=1:preset=veryfast:qp=0"
         "-x265-params":"crf="+str(local_q)+\
             ":keyint="+str(Quantization_H265.keyint)+":no-info=1"
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
        bpp = file_size*8.0/(h*w*Quantization_H265.scale_times*Quantization_H265.scale_times*GlobalVar.get_Temporal_LEN())
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
        # if GlobalVar.get_Istrain():
            
        # else:
        #     return decoded_frames,torch.autograd.Variable(torch.Tensor([bpp]))
        #     # return decoded_frames,torch.autograd.Variable(torch.Tensor([bpp]).cuda(output.device))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
import random
class Quantization_H265(nn.Module):
    def __init__(self,q = 17,keyint = 12,scale_times = 2):
        super(Quantization_H265, self).__init__()
        Quantization_H265.q = q
        Quantization_H265.keyint = keyint
        Quantization_H265.scale_times = scale_times
        Quantization_H265.file_random_name = None
    def forward(self, input):
        if not Quantization_H265.file_random_name:
            Quantization_H265.file_random_name = str(time.time())
        decodec_f =  Quant.apply(input)
        return decodec_f, (input - decodec_f.detach()) **2
