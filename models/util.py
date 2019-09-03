import torch.nn as nn
import torch
from transformation_measure import ObservableLayersModel

class Flatten(nn.Module):
    def forward(self, input:torch.Tensor):
        return input.view(input.size(0), -1)

class SequentialWithIntermediates(nn.Sequential,ObservableLayersModel):
    def __init__(self,*args):
        super(SequentialWithIntermediates, self).__init__(*args)

    def forward_intermediates(self,input_tensor)->(object,[]):
        outputs=[]
        for module in self._modules.values():
            input_tensor= module(input_tensor)
            outputs.append(input_tensor)
        return input_tensor,outputs

    def activation_names(self):
        outputs = []
        for module in self._modules.values():
            outputs.append(str(module))
        return outputs