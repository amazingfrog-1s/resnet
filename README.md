# SmallSamplesProblem
```
!cp "/content/drive/My Drive/FireDetection.zip" "/content/"
!unzip "/content/FireDetection.zip"
!cp "/content/drive/My Drive/COVID-19CT.zip" "/content/"
!unzip "/content/COVID-19CT.zip"
!cp "/content/drive/My Drive/resnet18.pth" "/content/"
!cp "/content/drive/My Drive/resnet34.pth" "/content/"
!cp "/content/drive/My Drive/resnet50.pth" "/content/"
!cp "/content/drive/My Drive/resnet101.pth" "/content/"
!cp "/content/drive/My Drive/resnet152.pth" "/content/"
!pip install tensorboardX

!python train.py --arch resnet152 --model /content/resnet152.pth --dataset /content/FireDetection/Train

!python train.py --arch resnet18 --model /content/resnet18.pth --dataset /content/COVID-19CT/Train

```


## using server to compute
```
python train.py --arch resnet18 --model ./resnet18.pth --dataset ../datasets/COVID-19CT/Train/


```
# resnet
# resnet
