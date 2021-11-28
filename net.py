from torchsummary import summary
from torchvision.models import resnet50
from torch import nn


net = nn.Sequential(
    nn.Conv2d(3,3,3,padding=1),
    resnet50(),
    nn.Linear(1000,100),
    nn.Linear(100,7)
)
print(net)

# summary(net().cuda(),input_size=(1,512,512))
# summary(generator().cuda(),input_size=(1,192,192))