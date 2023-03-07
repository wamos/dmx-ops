import sys, os
import torch
from torch import nn
import numpy as np
import time
from torch.profiler import profile, record_function, ProfilerActivity

from dmx_models import concat_cast_flatten, image_resize, reshape_casting, mel_scale

def run_dmx_ops(input_shape, name, num_threads):    
    input_var = torch.randn(*input_shape)
    device = torch.device("cpu")
    torch.set_num_threads(num_threads)
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
    iterations = 100
    time_list = np.zeros(iterations)
    for i in range(iterations):
        start = time.time()
        output = model(input) 
        end = time.time()
        time_list[i] = end - start
    print(np.median(time_list))
    # start = time.time()
    # iterations = 10
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         for x in range(iterations):
    #             output = model(input) 
    #         #print(output.shape)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # end = time.time()
    # avg_time = (end - start)/iterations
    # print(f"mean_data_transformation_time:{avg_time}")

if __name__ == '__main__':
    benchmark_name = sys.argv[1]
    num_threads = int(sys.argv[2])
    device = torch.device("cpu")
    if benchmark_name == "mel_scale":
		# actual size (1024, 768), batch=32
        #run_dmx_ops((32,1024,768), "mel_scale", num_threads)
        run_dmx_ops((32,1024,768), "mel_scale", num_threads)
        # run_dmx_ops((16,1024,768), "mel_scale", num_threads)
        # run_dmx_ops((8,1024,768), "mel_scale", num_threads)
        # run_dmx_ops((4,1024,768), "mel_scale", num_threads)
    elif benchmark_name == "reshape_casting":
		# actual size (128, 256), local_batch=128, global-batch=1024
        run_dmx_ops((1024, 128, 256), "reshape_casting", num_threads)
    elif benchmark_name== "image_resize":
		# actual size (1, 3, 1024, 768), batch=32
        run_dmx_ops((1, 32, 1024, 768), "image_resize", num_threads)
    elif benchmark_name== "concat_cast_flatten_aes":
		# actual size (1, 128, 256, 32), batch=32
        run_dmx_ops((1, 128, 768, 32), "concat_cast_flatten", num_threads)
    elif benchmark_name== "concat_cast_flatten_gzip":
		# actual size (1, 32, 1024, 512)
        run_dmx_ops((1, 64, 512, 512), "concat_cast_flatten", num_threads) 