from torch.nn import Module
from torch import nn
import torch


cnt = 0
def print_shape( x):
        global cnt
        print (f"the shape at {cnt} is {x.shape}")
        cnt += 1

# taken from benchmarks/model_generator.py: create_dmx_ops() function
class mel_scale(nn.Module):
    def __init__(self):
        super(mel_scale, self).__init__()
        
    def forward(self, x):
        #xt = torch.transpose(x,1,2)
        x = torch.flatten(x,1)
        y = torch.pow(x,2)
        y = torch.mul(y,0.001)
        y = torch.add(y,1)
        y = torch.tanh(y) # replace log with tanh
        y = torch.mul(y,2595) 
        y = y.type(torch.CharTensor)
        return y

class reshape_casting(nn.Module):
    def __init__(self):
        super(reshape_casting, self).__init__()

    def forward(self, x):
        y = torch.pow(x,2)
        y = torch.mul(x,0.5) # normalization constant                                
        #yt = torch.transpose(y,1,2)
        y = torch.reshape(y, (1024*8, 4096))
        y = y.type(torch.CharTensor)        
        return y

class image_resize(nn.Module):
    def __init__(self):
        super(image_resize, self).__init__()            
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, xt):
        #y  = nn.Identity(x)
        xt = torch.transpose(xt,2,3)
        z = self.max_pool(xt)
        return z

class concat_cast_flatten(nn.Module):
        def __init__(self):
            super(concat_cast_flatten, self).__init__() 

        def forward(self, x):       
            xc = torch.cat((x, x), 1)
            xt = xc.type(torch.IntTensor)
            xt = torch.div(xc,2)	
            xt = torch.transpose(xt,2,3)		
            xf = torch.flatten(xt,1)
            return xf
