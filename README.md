# In this repo we will learn the 101 walkthrough method to Finetune a Pretrained model from PyTorch library

### First of all we need a pretrained model from PyTorch library like ResNet18 because it's realatively small and we can modify easily

```
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```
###  First lets take a look at the structure of teh network and analyze how will our MNIST which is (28,28,1) go through the architecture.

### One important details is we need to turn off the parameters updates for the network -> we onlu want some parameters to be updated while training not the whole network. 

```
for param in resnet18_model.parameters():
  param.requires_grad = False
```

### We have two problems out here , channels in my data is gray and network takes 3 channels , so we need to modify the network layers and the second problem we need to solve is to take away the last layer and insert our MLP layer instead and make it learnable

```
resnet18_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

resnet18_model.fc = nn.Sequential (
    nn.Linear(num_features,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,10), # -> len(labels) labels according to labels
)
```

### Now let's take a look at how many learnable parameters do we have 

```
# Count the number of learnable parameters
total_params = sum(p.numel() for p in resnet18_model.parameters() if p.requires_grad)

print("Total number of learnable parameters:", total_params)
```

### And the rest is classical steps for preparing the data and doing the training and predictions later on

### I trained for only 1 epoch to make my POC valid -> you can increase the number of epochs and include more stuff like lr_scheduler depending on your dataset and use case