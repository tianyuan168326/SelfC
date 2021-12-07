import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os
import time
from global_var import GlobalVar


class H265_xxx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, original_input,DNN_output,q):
        input = original_input.clone().detach()
        dev_id = str(input.device)
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() 
        bt,c,h,w =output.size()
        frames = output.permute(0,2,3,1)
        frames = frames.cpu().numpy().astype(np.uint8)
        video_name = "./tmp/outputvideo"+dev_id+Quantization_H265_Suggrogate.file_random_name+"cvb.h265"
        



        p = {
            '-c:v': 'libx265',
            '-preset':'veryfast',
            '-tune':'zerolatency',
            "-s":str(w)+"x"+str(h),
            '-pix_fmt': 'yuv444p',
            "-vframes":"100",
            "-x265-params":"crf="+str(q)+\
                ":keyint="+str(Quantization_H265_Suggrogate.keyint)+":no-info=1"
        }
        import time
        T1 = time.time()
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
            decoded_frames += [torch.from_numpy(frame)]
        # print('runing time2 %s ms' % ((T3 - T2)*1000))
        decoded_frames = torch.stack(decoded_frames,dim=0).cuda(input.device)
        decoded_frames = decoded_frames.permute(0,3,1,2)
        decoded_frames = decoded_frames/255.
        # return output/255.
        ctx.save_for_backward(DNN_output,original_input,decoded_frames)
        
        return decoded_frames
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # print(grad_output)
    #     # print(grad_output.size())

    #     DNN_output,original_input,codec_output = ctx.saved_tensors
    #     DNN_g = torch.autograd.grad(
    #     DNN_output, original_input, grad_outputs=grad_output,retain_graph=True)[0]
    #     print(DNN_g.mean(),DNN_g.max(),DNN_g.min())
    #     print("grad_output",grad_output.mean(),grad_output.max(),grad_output.min())
    #     # DNN_g_real = torch.autograd.grad(
    #     # [DNN_output], [original_input], grad_outputs=torch.ones_like(DNN_output),retain_graph=True)[0]
    #     # torch.autograd.grad(codec_output,original_input,torch.ones_like(codec_output))
    #     # print(DNN_g.mean(),grad_output.mean())
    #     # grad_output = grad_output * DNN_g
    #     return DNN_g,None
        
# def h265_xxx( input):
    
   
import random
from models.modules.Subnet_constructor import *
class Quantization_H265_Suggrogate(nn.Module):
    def __init__(self,lambda_corr,q = 17,keyint = 12,scale_times = 2):
        super(Quantization_H265_Suggrogate, self).__init__()
        Quantization_H265_Suggrogate.q = q
        self.lambda_corr = lambda_corr
        Quantization_H265_Suggrogate.keyint = keyint
        Quantization_H265_Suggrogate.scale_times = scale_times
        Quantization_H265_Suggrogate.file_random_name = None
        mid_c = 24
        self.suggrogate_net = nn.Sequential(
            DenseBlock(4,mid_c,INN_init=False),
            DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            # FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            # FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            # FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            # FeatureCalapseBlock(mid_c,mid_c,INN_init=True,is_res=True),
            # DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            # DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            DenseBlock(mid_c,mid_c,INN_init=False,is_res=True),
            DenseBlock(mid_c,3,INN_init=False),
        )
        if isinstance(Quantization_H265_Suggrogate.q, list):
            self.indicator_fuser = nn.Sequential(
                nn.Linear(2,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1),
            )
    def forward(self, input):
        if not Quantization_H265_Suggrogate.file_random_name:
            Quantization_H265_Suggrogate.file_random_name = str(time.time())
        bt,c,h,w = input.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt//t
        input_3d = input.reshape(b,t,c,h,w)
        #### fixed Q, we pad the time stamp
        if isinstance(Quantization_H265_Suggrogate.q, int):
            temporal_indicator = torch.linspace(0,1,t).cuda(input_3d.device)
            indicator = temporal_indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\
                .repeat(b,1,1,h,w)
            current_q = Quantization_H265_Suggrogate.q
        elif isinstance(Quantization_H265_Suggrogate.q, list):
            current_q =  random.randint(Quantization_H265_Suggrogate.q[0],Quantization_H265_Suggrogate.q[1]) 
            ### dynamic Q, we fuse (Q and temporal) as a token
            temporal_indicator = torch.linspace(0,1,t).cuda(input_3d.device) # t
            q_indicator = torch.zeros_like(temporal_indicator).cuda(input_3d.device).fill_(current_q/30) #t
            indicator = torch.stack([temporal_indicator,q_indicator],dim=1) ## 3 x 2
            # print(indicator.size(),"indicator")
            indicator = self.indicator_fuser(indicator.unsqueeze(0))### 1 3 1  b t c(hw)
            indicator = indicator.unsqueeze(-1).unsqueeze(-1)\
                .repeat(b,1,1,h,w)
            
        input_3d_temporal_ind = torch.cat([input_3d,indicator],dim=2)

        sug_out = self.suggrogate_net(input_3d_temporal_ind.reshape(bt,c+1,h,w))\
            +input
        H265_encoder_encoder_out = H265_xxx.apply(input,sug_out,current_q)
        x = H265_encoder_encoder_out.detach()
        y = sug_out
        # print(x.size())
        # exit(0)
        mimick_loss = torch.mean((x - y)**2.0)

        vx = x - torch.mean(x,dim=0,keepdim=True)
        vy = y - torch.mean(y,dim=0,keepdim=True)

        correlation_param = torch.sum(vx * vy,dim=0,keepdim=True) / \
             (torch.sqrt(torch.sum(vx ** 2,dim=0,keepdim=True)) * torch.sqrt(torch.sum(vy ** 2,dim=0,keepdim=True))+1e-8)
        correlation_param = correlation_param.mean()
        # print(correlation_param,mimick_loss,correlation_param.size())
        sug_out.data = H265_encoder_encoder_out
        return sug_out,mimick_loss - self.lambda_corr*correlation_param
    
