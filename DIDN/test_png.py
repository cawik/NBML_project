import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from matplotlib import pyplot as plt
import torch.nn as nn
import imageio
from gray_model import _NetG
import sys

parser = argparse.ArgumentParser(description="PyTorch DIDN  Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/pretrained_gray/gray_model.pth", type=str, help="model path")
parser.add_argument("--output_path", default="./result/result_images/", type=str, help="output path")
parser.add_argument("--self_ensemble", action="store_true", help="Use self ensemble?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = _NetG()
checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'].state_dict())

from skimage import io as skimage_io

#im_bmp = imageio.imread('noised_frame_1.bmp')
#im = im_bmp.astype(float) / 255.

im = skimage_io.imread('noised_frame_1.png', as_gray=True)

#sys.exit()

noisy = np.array(im)
origin = noisy.copy()

avg_psnr_predicted = 0.0
avg_psnr_noisy = 0.0
avg_elapsed_time = 0.0
ct = 0.0

print("print noisy\n", noisy.shape, type(noisy), "\n", noisy)
psnr_noisy = output_psnr_mse(origin, noisy)
avg_psnr_noisy += psnr_noisy
noisy_data = []
out_data = []
output = 0

with torch.no_grad():
    model.eval()
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    if opt.self_ensemble:
        # rotate / flip
        noisy_data.append(noisy)
        noisy_data.append(np.rot90(noisy, 1).copy())
        noisy_data.append(np.rot90(noisy, 2).copy())
        noisy_data.append(np.rot90(noisy, 3).copy())
        noisy_data.append(np.fliplr(noisy_data[0]).copy())
        noisy_data.append(np.fliplr(noisy_data[1]).copy())
        noisy_data.append(np.fliplr(noisy_data[2]).copy())
        noisy_data.append(np.fliplr(noisy_data[3]).copy())

        for x in range(8):
            print("SIZE:\n", noisy.shape, "\n")
            
            noisy = Variable(torch.from_numpy(noisy_data[x]).float()).view(1, 1, noisy_data[x].shape[0], noisy_data[x].shape[1])
            noisy = noisy.cuda()

            start_time = time.time()
            out = model(noisy)

            out_data.append(out[0, 0, :, :].cpu().detach().numpy().astype(np.float32))
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

        out_data[4] = np.fliplr(out_data[4])
        out_data[5] = np.fliplr(out_data[5])
        out_data[6] = np.fliplr(out_data[6])
        out_data[7] = np.fliplr(out_data[7])

        out_data[1] = np.rot90(out_data[1], -1)
        out_data[2] = np.rot90(out_data[2], -2)
        out_data[3] = np.rot90(out_data[3], -3)

        out_data[5] = np.rot90(out_data[5], -1)
        out_data[6] = np.rot90(out_data[6], -2)
        out_data[7] = np.rot90(out_data[7], -3)

        for x in range(8):
            output += out_data[x]
        output /= 8.0

    else:  # no self ensemble
        noisy = Variable(torch.from_numpy(noisy).float()).view(1, 1, noisy.shape[0], noisy.shape[1])
        noisy = noisy.cuda()

        start_time = time.time()
        out = model(noisy)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        output = out.cpu()
        output = output.data[0].numpy().astype(np.float32)
        output = output[0,:,:]

    output[output>1] = 1
    output[output<0] = 0

    psnr_predicted = output_psnr_mse(origin, output)
    avg_psnr_predicted += psnr_predicted
    ct += 1

    output = output * 255.
    output = np.uint8(np.round(output))

    test_name = "denoised_frame_1_no_ens" + '.png'  # .mat -> .png
    imageio.imwrite(opt.output_path + test_name, output)  # save result images


avg_psnr_predicted = avg_psnr_predicted / ct
avg_psnr_noisy = avg_psnr_noisy / ct

print("PSNR_noisy=", avg_psnr_noisy)
print("PSNR_predicted=", avg_psnr_predicted)

