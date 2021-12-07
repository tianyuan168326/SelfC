import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import torchvision
PATH = "/dev/shm/"
# PATH = "./"
import time
# libpng_path_enc = "/media/ps/SSD/tianyuan/libbpg-0.9.8/bpgenc"
# libpng_path_dec = "/media/ps/SSD/tianyuan/libbpg-0.9.8/bpgdec"

libpng_path_enc = "/data_video/libbpg-0.9.8/bpgenc"
libpng_path_dec = "/data_video/libbpg-0.9.8/bpgdec"


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        dev_id = str(input.device)
        input = torch.clamp(input, 0, 1)
        quant_img = (input * 255.).round()/255.0
        quant_error = torch.mean((quant_img - input)**2)
        print("quant_error",quant_error)
        # quant_img = input
        img_num = quant_img.size(0)
        new_imgs = []
        bpps = []
        for img_i in range(img_num):
            now_img = quant_img[img_i]
            c,h,w = now_img.size()
            img_name = dev_id +"_"+ str(img_i)
            png_path = PATH+img_name+"_old.png"
            # Image.fromarray(now_img.permute(1,2,0).cpu().numpy(),"RGB").save(png_path)
            # print("now_img",now_img.size())
            torchvision.utils.save_image(now_img,png_path)
            # transforms.ToPILImage()(now_img.cpu()).save(png_path)
            bpg_name = PATH+img_name+".bpg"
            convert_png_bpg = libpng_path_enc+" -q 20 -o " +bpg_name +" "+png_path
            os.system(convert_png_bpg)
            file_size = os.path.getsize(bpg_name)
            bpp = file_size*8/(h*w*2*2)
            bpps += [bpp]
            png_path = PATH+img_name+"_new.png"
            convert_bpg_png = libpng_path_dec+" -o "+png_path +" "+ bpg_name
            os.system(convert_bpg_png)
            new_img =  torch.from_numpy(np.asarray(Image.open(png_path))).cuda(input.device).permute(2,0,1)
            new_imgs += [new_img]
        new_imgs = torch.stack(new_imgs,dim=0)

        
        # n,c,h,w = input.size()
        # now_big_img = input.permute(1,0,2,3).reshape(c,n*h,w)
        # img_name = dev_id
        # png_path = PATH+img_name+"_old.png"
        # # print("t1",dev_id, time.time())
        # torchvision.utils.save_image(now_big_img,png_path)
        # # transforms.ToPILImage()(now_img.cpu()).save(png_path)
        # bpg_path = PATH+img_name+".bpg"
        # convert_png_bpg = libpng_path_enc+" -q 20 -o " +bpg_path +" "+png_path
        # # print("t2",dev_id, time.time())

        # os.system(convert_png_bpg)
        # png_path = PATH+img_name+"_new.png"
        # convert_bpg_png = libpng_path_dec+" -o "+png_path +" "+ bpg_path
        # # print("t3",dev_id, time.time())

        # os.system(convert_bpg_png)
        # # print("t4",dev_id, time.time())
        # new_big_img =  torch.from_numpy(np.asarray(Image.open(png_path))).cuda(input.device).permute(2,0,1)
        # # print("t5",dev_id, time.time())
        # new_imgs = new_big_img.reshape(c,n,h,w).permute(1,0,2,3)

        output = new_imgs / 255.
        coding_err = torch.mean((output-quant_img)**2)
        print("coding_err",coding_err)
        # output = quant_img
        return output, torch.autograd.Variable(torch.Tensor([sum(bpps)/len(bpps)]).cuda(output.device))
        

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization_BPG(nn.Module):
    def __init__(self):
        super(Quantization_BPG, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
