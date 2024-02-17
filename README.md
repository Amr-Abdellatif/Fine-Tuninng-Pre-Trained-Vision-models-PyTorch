# In this repo we will learn the 101 walkthrough method to Finetune a Pretrained model from PyTorch library

## First of all we need a pretrained model from PyTorch library like ResNet18 because it's realatively small and we can modify easily

```
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

