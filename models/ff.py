from torch import nn
from transformation_measure import ObservableLayersModel
from models import SequentialWithIntermediates

class FFNet(nn.Module,ObservableLayersModel):
    def __init__(self,input_shape,num_classes,h1=128,h2=64):
        super(FFNet, self).__init__()
        self.name = self.__class__.__name__

        h,w,channels=input_shape

        self.linear_size = h * w * channels
        self.fc= SequentialWithIntermediates(
                nn.Linear(self.linear_size, h1),
               nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
               nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Linear(h2, num_classes),
                nn.LogSoftmax()
                )

    def forward(self, x):
        x = x.view(-1, self.linear_size)
        x= self.fc(x)
        return x

    def activation_names(self):
        return ["fc1","bn1","fc1act","fc2","bn2","fc2act","fc3","logsoft"]

    def forward_intermediates(self,x)->(object,[]):
        x = x.view(-1, self.linear_size)
        x,intermediates = self.fc.forward_intermediates(x)
        return x,intermediates

