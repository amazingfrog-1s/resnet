#coding:utf-8
import sys
from PIL import Image
from torchvision import transforms
import glob as gb

#img_read_path = gb.glob("D:/Datasets/SmallData2020/COVID-19CT/Train/positive/*.png")

#img_read_path = gb.glob("D:/Datasets/SmallData2020/COVID-19CT/Train/negetive/*.png")

img_read_path = gb.glob("D:/Datasets/SmallData2020/COVID-19CT/Train/negetive/*.jpg")

#img_save_path = "D:/Datasets/SmallData2020/covid19add/positive/"

img_save_path = "D:/Datasets/SmallData2020/covid19add/negetive/"

counter = 0
for path in img_read_path:
    for i in range(10):
        img = Image.open(path)
        img = transforms.Resize((256,256))(img)
        img = transforms.RandomRotation(15)(img)
        img = transforms.RandomCrop(240)(img)
        img = transforms.CenterCrop(224)(img)

        img.save(img_save_path + str(counter) +".png")
        counter = counter + 1


# Resize
#transforms.Resize()

# Random Crop
#transforms.RandomCrop()(img)

# Center Crop
#transforms.CenterCrop()(img)

# H Flip
#transforms.RandomHorizontalFlip()(img)

#
#transforms.RandomVerticalFlip()(img)

#
#tranforms.RandomRotation(30)(img)

#
#transforms.ColorJitter(brightness=1)(img)

#transforms.ColorJitter(contrast=1)(img)

#transforms.ColorJitter(saturation=1)(img)

#transforms.ColorJitter(hue=0.5)(img)






