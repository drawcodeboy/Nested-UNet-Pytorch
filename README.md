# UNet++(Nested UNet) Implementation in PyTorch
<img src="https://img.shields.io/badge/PyTorch 2.0.1-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

```
from Nested_UNet import Nested_UNet

model = Nested_UNet(channel=1, mode='accurate')
```

* * *
# Configuration
![image](https://blog.kakaocdn.net/dn/k94V4/btqDFo3FuBW/9py6IMKOdNQWWe2vdmF8Kk/img.png)
* * *
# torchsummary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 224, 224]             320
       BatchNorm2d-2         [-1, 32, 224, 224]              64
              ReLU-3         [-1, 32, 224, 224]               0
            Conv2d-4         [-1, 32, 224, 224]           9,248
       BatchNorm2d-5         [-1, 32, 224, 224]              64
              ReLU-6         [-1, 32, 224, 224]               0
         ConvBlock-7         [-1, 32, 224, 224]               0
         MaxPool2d-8         [-1, 32, 112, 112]               0
      EncoderBlock-9  [[-1, 32, 224, 224], [-1, 32, 112, 112]]               0
           Conv2d-10         [-1, 64, 112, 112]          18,496
      BatchNorm2d-11         [-1, 64, 112, 112]             128
             ReLU-12         [-1, 64, 112, 112]               0
           Conv2d-13         [-1, 64, 112, 112]          36,928
      BatchNorm2d-14         [-1, 64, 112, 112]             128
             ReLU-15         [-1, 64, 112, 112]               0
        ConvBlock-16         [-1, 64, 112, 112]               0
        MaxPool2d-17           [-1, 64, 56, 56]               0
     EncoderBlock-18  [[-1, 64, 112, 112], [-1, 64, 56, 56]]               0
           Conv2d-19          [-1, 128, 56, 56]          73,856
      BatchNorm2d-20          [-1, 128, 56, 56]             256
             ReLU-21          [-1, 128, 56, 56]               0
           Conv2d-22          [-1, 128, 56, 56]         147,584
      BatchNorm2d-23          [-1, 128, 56, 56]             256
             ReLU-24          [-1, 128, 56, 56]               0
        ConvBlock-25          [-1, 128, 56, 56]               0
        MaxPool2d-26          [-1, 128, 28, 28]               0
     EncoderBlock-27  [[-1, 128, 56, 56], [-1, 128, 28, 28]]               0
           Conv2d-28          [-1, 256, 28, 28]         295,168
      BatchNorm2d-29          [-1, 256, 28, 28]             512
             ReLU-30          [-1, 256, 28, 28]               0
           Conv2d-31          [-1, 256, 28, 28]         590,080
      BatchNorm2d-32          [-1, 256, 28, 28]             512
             ReLU-33          [-1, 256, 28, 28]               0
        ConvBlock-34          [-1, 256, 28, 28]               0
        MaxPool2d-35          [-1, 256, 14, 14]               0
     EncoderBlock-36  [[-1, 256, 28, 28], [-1, 256, 14, 14]]               0
           Conv2d-37          [-1, 512, 14, 14]       1,180,160
      BatchNorm2d-38          [-1, 512, 14, 14]           1,024
             ReLU-39          [-1, 512, 14, 14]               0
           Conv2d-40          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-41          [-1, 512, 14, 14]           1,024
             ReLU-42          [-1, 512, 14, 14]               0
        ConvBlock-43          [-1, 512, 14, 14]               0
        MaxPool2d-44            [-1, 512, 7, 7]               0
     EncoderBlock-45  [[-1, 512, 14, 14], [-1, 512, 7, 7]]               0
  ConvTranspose2d-46         [-1, 32, 224, 224]           8,224
           Conv2d-47         [-1, 32, 224, 224]          18,464
      BatchNorm2d-48         [-1, 32, 224, 224]              64
             ReLU-49         [-1, 32, 224, 224]               0
           Conv2d-50         [-1, 32, 224, 224]           9,248
      BatchNorm2d-51         [-1, 32, 224, 224]              64
             ReLU-52         [-1, 32, 224, 224]               0
        ConvBlock-53         [-1, 32, 224, 224]               0
     DecoderBlock-54         [-1, 32, 224, 224]               0
  ConvTranspose2d-55         [-1, 64, 112, 112]          32,832
           Conv2d-56         [-1, 64, 112, 112]          73,792
      BatchNorm2d-57         [-1, 64, 112, 112]             128
             ReLU-58         [-1, 64, 112, 112]               0
           Conv2d-59         [-1, 64, 112, 112]          36,928
      BatchNorm2d-60         [-1, 64, 112, 112]             128
             ReLU-61         [-1, 64, 112, 112]               0
        ConvBlock-62         [-1, 64, 112, 112]               0
     DecoderBlock-63         [-1, 64, 112, 112]               0
  ConvTranspose2d-64          [-1, 128, 56, 56]         131,200
           Conv2d-65          [-1, 128, 56, 56]         295,040
      BatchNorm2d-66          [-1, 128, 56, 56]             256
             ReLU-67          [-1, 128, 56, 56]               0
           Conv2d-68          [-1, 128, 56, 56]         147,584
      BatchNorm2d-69          [-1, 128, 56, 56]             256
             ReLU-70          [-1, 128, 56, 56]               0
        ConvBlock-71          [-1, 128, 56, 56]               0
     DecoderBlock-72          [-1, 128, 56, 56]               0
  ConvTranspose2d-73          [-1, 256, 28, 28]         524,544
           Conv2d-74          [-1, 256, 28, 28]       1,179,904
      BatchNorm2d-75          [-1, 256, 28, 28]             512
             ReLU-76          [-1, 256, 28, 28]               0
           Conv2d-77          [-1, 256, 28, 28]         590,080
      BatchNorm2d-78          [-1, 256, 28, 28]             512
             ReLU-79          [-1, 256, 28, 28]               0
        ConvBlock-80          [-1, 256, 28, 28]               0
     DecoderBlock-81          [-1, 256, 28, 28]               0
  ConvTranspose2d-82         [-1, 32, 224, 224]           8,224
           Conv2d-83         [-1, 32, 224, 224]          27,680
      BatchNorm2d-84         [-1, 32, 224, 224]              64
             ReLU-85         [-1, 32, 224, 224]               0
           Conv2d-86         [-1, 32, 224, 224]           9,248
      BatchNorm2d-87         [-1, 32, 224, 224]              64
             ReLU-88         [-1, 32, 224, 224]               0
        ConvBlock-89         [-1, 32, 224, 224]               0
     DecoderBlock-90         [-1, 32, 224, 224]               0
  ConvTranspose2d-91         [-1, 64, 112, 112]          32,832
           Conv2d-92         [-1, 64, 112, 112]         110,656
      BatchNorm2d-93         [-1, 64, 112, 112]             128
             ReLU-94         [-1, 64, 112, 112]               0
           Conv2d-95         [-1, 64, 112, 112]          36,928
      BatchNorm2d-96         [-1, 64, 112, 112]             128
             ReLU-97         [-1, 64, 112, 112]               0
        ConvBlock-98         [-1, 64, 112, 112]               0
     DecoderBlock-99         [-1, 64, 112, 112]               0
 ConvTranspose2d-100          [-1, 128, 56, 56]         131,200
          Conv2d-101          [-1, 128, 56, 56]         442,496
     BatchNorm2d-102          [-1, 128, 56, 56]             256
            ReLU-103          [-1, 128, 56, 56]               0
          Conv2d-104          [-1, 128, 56, 56]         147,584
     BatchNorm2d-105          [-1, 128, 56, 56]             256
            ReLU-106          [-1, 128, 56, 56]               0
       ConvBlock-107          [-1, 128, 56, 56]               0
    DecoderBlock-108          [-1, 128, 56, 56]               0
 ConvTranspose2d-109         [-1, 32, 224, 224]           8,224
          Conv2d-110         [-1, 32, 224, 224]          36,896
     BatchNorm2d-111         [-1, 32, 224, 224]              64
            ReLU-112         [-1, 32, 224, 224]               0
          Conv2d-113         [-1, 32, 224, 224]           9,248
     BatchNorm2d-114         [-1, 32, 224, 224]              64
            ReLU-115         [-1, 32, 224, 224]               0
       ConvBlock-116         [-1, 32, 224, 224]               0
    DecoderBlock-117         [-1, 32, 224, 224]               0
 ConvTranspose2d-118         [-1, 64, 112, 112]          32,832
          Conv2d-119         [-1, 64, 112, 112]         147,520
     BatchNorm2d-120         [-1, 64, 112, 112]             128
            ReLU-121         [-1, 64, 112, 112]               0
          Conv2d-122         [-1, 64, 112, 112]          36,928
     BatchNorm2d-123         [-1, 64, 112, 112]             128
            ReLU-124         [-1, 64, 112, 112]               0
       ConvBlock-125         [-1, 64, 112, 112]               0
    DecoderBlock-126         [-1, 64, 112, 112]               0
 ConvTranspose2d-127         [-1, 32, 224, 224]           8,224
          Conv2d-128         [-1, 32, 224, 224]          46,112
     BatchNorm2d-129         [-1, 32, 224, 224]              64
            ReLU-130         [-1, 32, 224, 224]               0
          Conv2d-131         [-1, 32, 224, 224]           9,248
     BatchNorm2d-132         [-1, 32, 224, 224]              64
            ReLU-133         [-1, 32, 224, 224]               0
       ConvBlock-134         [-1, 32, 224, 224]               0
    DecoderBlock-135         [-1, 32, 224, 224]               0
          Conv2d-136          [-1, 1, 224, 224]              33
          Conv2d-137          [-1, 1, 224, 224]              33
          Conv2d-138          [-1, 1, 224, 224]              33
          Conv2d-139          [-1, 1, 224, 224]              33
         Sigmoid-140          [-1, 1, 224, 224]               0
================================================================
Total params: 9,048,996
Trainable params: 9,048,996
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 4521.27
Params size (MB): 34.52
Estimated Total Size (MB): 4555.98
----------------------------------------------------------------
```
# Reference
Paper 1: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
