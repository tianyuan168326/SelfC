import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os
from global_var import GlobalVar
import time

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

class Quantization_H265_Stream:
    def __init__(self,q = 17,keyint = 12,scale_times = 2,opt = None):
        self.q = q
        self.keyint = keyint
        self.scale_times = scale_times
        Quantization_H265_Stream.file_random_name = None
        self.h265_all_default = opt["h265_all_default"]
        self.video_frame_num = 0

    def open_writer(self,dev_id,w,h ):
        if not Quantization_H265_Stream.file_random_name:
            Quantization_H265_Stream.file_random_name = str(time.time())
        self.w = w
        self.h = h
        self.video_frame_num = 0
        self.video_name = "./tmp/outputvideo"+str(dev_id)+ Quantization_H265_Stream.file_random_name+"cvb.mkv"
        # self.video_name = "./tmp/32E2R.h265"
        # self.p = {
        # '-c:v': 'libx265',
        # '-preset':'slow',
        # '-tune':'zerolatency',
        # "-s":str(w)+"x"+str(h),
        # '-pix_fmt': 'yuv420p',
        # # "-vframes":"100",
        # # "-r": "60",
        #  "-x265-params":"crf="+str(self.q)+\
        #      ":keyint="+str(self.keyint)+":no-info=1:subme=1:ref=2:rd=2:early-skip=1:fast-intra=1:lookahead-slices=4:preset=slow:level-idc=4:no-high-tier=1"
        # }
        # ##:pools=none:numa-pools=none
        # self.writer = skvideo.io.FFmpegWriter(self.video_name,
        # inputdict = {
        #     "-s":str(w)+"x"+str(h),
        #     "-pix_fmt":"yuv420p",
        # },
        # outputdict = self.p,verbosity = 1)
        if self.keyint and self.keyint>0:
            x265_config = "crf={}:keyint={}:no-info=1".format(self.q,self.keyint)
        else:
            x265_config = "crf={}:no-info=1".format(self.q)
        out_dict  ={
            "-s":str(w)+"x"+str(h),
            "-pix_fmt":"yuv444p",
            "-c:v":"libx265", 
            "-preset":"veryfast",
            "-tune":"zerolatency",
            "-x265-params":x265_config
        }
        if self.h265_all_default:
            out_dict  ={
            "-s":str(w)+"x"+str(h),
            "-pix_fmt":"yuv444p",
            "-c:v":"libx265",
            "-x265-params":x265_config
            }
        self.rgb_yuv_writter = skvideo.io.FFmpegWriter(self.video_name,
        inputdict = {
            "-s":str(w)+"x"+str(h),
            "-pix_fmt":"rgb24",
        },
        outputdict = out_dict,verbosity = 1)
    def write_multi_frames(self,input):
        input = torch.clamp(input, 0.0, 1.0)
        output = (input * 255.0).round()### b c h w 
        bt,c,h,w =output.size()
        self.h = h
        self.w = w
        frames = output.permute(0,2,3,1)
        frames = frames.cpu().numpy().astype(np.uint8)
        for i in range(bt):
            self.video_frame_num = self.video_frame_num+1
            self.rgb_yuv_writter.writeFrame(frames[i, :, :, :])
        
    def close_writer(self):
    ### return bpp
        self.rgb_yuv_writter.close()
        # print(self.video_name)
        # exit()
        # print("begin h265 encode")
        # cmd = '''ffmpeg -y -pix_fmt yuv420p -s {}x{} -c:v rawvideo -r 60 -i {} -c:v libx265  -preset veryfast -tune zerolatency  -x265-params "crf={}:keyint={}:no-info=1" {}'''.format(
        #     str(self.w),
        #     str(self.h),
        #     self.tmp_yuv_name,
        #     self.q,
        #     self.keyint,
        # )
        # cmd +=" -hide_banner -loglevel error"
        # os.system(cmd)
        # print("begin h265 decode")
        # # print(cmd)

   
        file_size = os.path.getsize(self.video_name)
        print("self.video_frame_num",self.video_frame_num)
        print("\n\n")
        bpp = file_size*8.0/(self.h*self.w*self.scale_times*self.scale_times*self.video_frame_num)
        self.video_frame_num = 0
        print("bpp",bpp)
        # exit(0)
        return torch.autograd.Variable(torch.Tensor([bpp]))
    def open_reader(self):
        outputparameters = {}
        print("begin read")
        self.reader = skvideo.io.FFmpegReader(self.video_name,
                        inputdict= {
                            # "-s":str(self.w)+"x"+str(self.h),
                            # "-pix_fmt":"yuv420p",
                            # "-c:v":"libx265"
                        },
                        outputdict={
                            # "-s":str(self.w)+"x"+str(self.h),
                        },verbosity  =1)
    def read_multi_frames(self,num):
        decoded_frames = []
        count = 0
        for frame in self.reader.nextFrame():
            # do something with the ndarray frame
            # decoded_frames += [torch.from_numpy(frame).cuda(input.device)]
            decoded_frames += [torch.from_numpy(frame.astype(np.float32)/255.0)]
            count = count+1
            if count == num:
                break

        decoded_frames = torch.stack(decoded_frames,dim=0)
        decoded_frames = decoded_frames.permute(0,3,1,2)
        decoded_frames = decoded_frames
        return decoded_frames
