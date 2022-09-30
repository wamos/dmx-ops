import sys, os
import torch
from torch import nn
import time

from dmx_models import concat_cast_flatten, image_resize, reshape_casting, mel_scale

def run_dmx_ops(input_shape, name):    
    input_var = torch.randn(*input_shape)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    input = input_var.to(device)
    if name == "mel_scale":
        model = mel_scale().to(device)
    elif name  == "reshape_casting":
        model = reshape_casting().to(device)
    elif name == "image_resize":
        model = image_resize().to(device)
    elif name == "concat_cast_flatten":
        model = concat_cast_flatten().to(device)

    model.eval()
    start = time.time()
    iterations = 10
    for x in range(iterations):
        output = model(input) 
        #print(output.shape)
    end = time.time()
    avg_time = (end - start)/iterations
    print(f"mean_data_transformation_time:{avg_time}")

if __name__ == '__main__':
    benchmark_name = sys.argv[1]
    device = torch.device("cpu")
    if benchmark_name == "mel_scale":
		# actual size (1024, 768), batch=32
        run_dmx_ops((32, 1024, 768), "mel_scale")
    elif benchmark_name == "reshape_casting":
		# actual size (128, 256), local_batch=128, global-batch=1024
        run_dmx_ops((1024, 128, 256), "reshape_casting")
    elif benchmark_name== "image_resize":
		# actual size (1, 3, 1024, 768), batch=32
        run_dmx_ops((1, 32, 1024, 768), "image_resize")
    elif benchmark_name== "concat_cast_flatten_aes":
		# actual size (1, 128, 256, 32), batch=32
        run_dmx_ops((1, 128, 768, 32), "concat_cast_flatten")
    elif benchmark_name== "concat_cast_flatten_gzip":
		# actual size (1, 32, 1024, 512)
        run_dmx_ops((1, 64, 512, 512), "concat_cast_flatten") 