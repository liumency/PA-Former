from torchvision.models import resnet18
from thop import profile
import torch
from model.network import CDNet

netCD = CDNet()

input1 = torch.randn(1, 3, 512, 512)
input2 = torch.randn(1, 3, 512, 512)
flops, params = profile(netCD, inputs=(input1, input2,))
print(flops/1e9, params/1e6) #flops单位G，para单位M
