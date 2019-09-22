import torch.nn as nn
import torch
from transformation_measure import ObservableLayersModule

import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input:torch.Tensor):
        return input.view(input.size(0), -1)

class SequentialWithIntermediates(nn.Sequential, ObservableLayersModule):
    def __init__(self,*args):
        super(SequentialWithIntermediates, self).__init__(*args)

    def forward_intermediates(self,input_tensor)->(object,[]):
        submodules=self._modules.values()
        if len(submodules)==0:
            return  input_tensor,[input_tensor]

        outputs=[]
        for module in submodules:
            if isinstance(module, ObservableLayersModule):
                input_tensor,intermediates=module.forward_intermediates(input_tensor)
                outputs+=(intermediates)
            else:
                input_tensor= module(input_tensor)
                outputs.append(input_tensor)
        return input_tensor,outputs

    def activation_names(self)->[str]:
        submodules = self._modules.values()
        if len(submodules) == 0:
            return ["identity"]
        if len(submodules) == 1:
            module = list(submodules)[0]

            if isinstance(module, ObservableLayersModule):
                return ["0_"+name for name in module.activation_names()]
            else:
                name = module.__class__.__name__
                return [self.abbreviation(name)]

        # len(submodules)>1
        outputs = []
        index=0

        for module in submodules:
            if isinstance(module, ObservableLayersModule):
                index += 1
                module_name=self.abbreviation(module.__class__.__name__)
                names=[f"{module_name}{index}_{name}" for name in module.activation_names()]
                outputs +=names
            else:
                name=module.__class__.__name__
                if name.startswith("Conv") or name.startswith("Linear"):
                    index += 1  # conv and fc layers increase index
                name=f"{index}{self.abbreviation(name)}"
                outputs.append(name)
        return outputs

    def abbreviation(self, name:str)->str:
        if name.startswith("Conv"):
            name = "c"
        elif name.startswith("BatchNorm"):
            name = "bn"
        elif name.startswith("ELU"):
            name = "elu"
        elif name.startswith("ReLU"):
            name= "relu"
        elif name.startswith("Linear"):
            name = "fc"
        elif name.startswith("Add"):
            name = "+"
        elif "Softmax" in name:
            name="sm"
        elif name == "Sequential":
            name = ""
        elif name == "SequentialWithIntermediates":
            name = ""
        elif name == "Block":
            name = "b"
        return name

class Add(ObservableLayersModule):

    def __init__(self, left:ObservableLayersModule, right:ObservableLayersModule):
        super(Add,self).__init__()
        self.left=left
        self.right=right

    def forward(self,x):
        left_x=self.left(x)
        right_x = self.right(x)
        return left_x+right_x

    def forward_intermediates(self,x):
        left_x,left_outputs = self.left.forward_intermediates(x)
        right_x,right_outputs = self.right.forward_intermediates(x)
        output=left_x + right_x
        return output,left_outputs+right_outputs+[output]

    def activation_names(self):
        left_names=self.left.activation_names()
        right_names = self.right.activation_names()

        right_last = right_names[-1]# if right_names else "right"
        left_last = left_names[-1]# if left_names else "left"
        this=[left_last+"+"+right_last]
        return left_names+right_names+this


class GlobalAvgPool2d(nn.Module):

    def forward(self,x):
        return F.adaptive_avg_pool2d(x,(1,1))