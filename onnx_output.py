import onnx
import sys, os
import numpy as np
import torch
import torch.onnx
from torch import nn
from dmx_models import image_resize, reshape_casting, mel_scale

if __name__ == '__main__':
	#onnx_name = sys.argv[1]
	onnx_name = "ppo_model.onnx"
	onnx_model = onnx.load(os.path.join('./', onnx_name))
	print(onnx_model)

    #oh_ow_dim0 = int(sys.argv[1])
    #oh_ow_dim1 = int(sys.argv[2])
    #layer = sys.argv[2]    
    # batch_size = 1

    
    # oh_ow_dim0 = 250
    # oh_ow_dim1 = 1 
    # model = casting_with_reshape()
    # train_dataset = torch.randn(oh_ow_dim0, oh_ow_dim1, requires_grad=True)

    # oh_ow_dim0=128
    # oh_ow_dim1=750
    # model = flatten()
    # train_dataset = torch.randn(1, oh_ow_dim0, oh_ow_dim1, requires_grad=True)

    # oh_ow_dim0=128
    # oh_ow_dim1=750
    # model = mel_scale()
    # train_dataset = torch.randn(1, oh_ow_dim0, oh_ow_dim1, requires_grad=True)

    # oh_ow_dim0=1080
    # oh_ow_dim1=720
    # model = avg_pool()
    # train_dataset = torch.randn(3, oh_ow_dim0, oh_ow_dim1, requires_grad=True)
        
    # torch.onnx.export(model,               # model being run
    #           train_dataset,                         # model input (or a tuple for multiple inputs)
    #           onnx_name,   # where to save the model (can be a file or file-like object)
    #           training = torch.onnx.TrainingMode.EVAL,
    #           keep_initializers_as_inputs=True,
    #           export_params=True,        # store the trained parameter weights inside the model file
    #           opset_version=10,          # the ONNX version to export the model to, default is 10 here
    #           do_constant_folding=True,  # whether to execute constant folding for optimization
    #           input_names = ['input'],   # the model's input names
    #           output_names = ['output']) # the model's output names
    #           #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #           #              'output' : {0 : 'batch_size'}})
    # onnx_model = onnx.load(os.path.join('./', onnx_name))
    # print(onnx_model)
