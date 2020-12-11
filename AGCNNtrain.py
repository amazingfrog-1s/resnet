#train
# encoding: utf-8
#!/usr/bin/env python3
# encoding: utf-8
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import argparse
__all__ = ['DenseNet', 'Densenet121_AG']

parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--dataset',required=True)
args = parser.parse_args()
CKPT_PATH = './'
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def Densenet121_AG(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
 
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
 
 
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
 
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
 
        super(DenseNet, self).__init__()
 
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
 
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
 
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
 
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        self.Sigmoid = nn.Sigmoid()
 
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out_after_pooling = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out_after_pooling)
        out = self.Sigmoid(out)
        return out, features, out_after_pooling


class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion_Branch, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        #fusion = torch.cat((global_pool.unsqueeze(2), local_pool.unsqueeze(2)), 2).cuda()
        #fusion = fusion.max(2)[0]#.squeeze(2).cuda()                                   
        #print(fusion.shape)
        fusion = torch.cat((global_pool,local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        x = self.fc(fusion_var)
        x = self.Sigmoid(x)

        return x


#========================================================================================================#
#    train    #
#========================================================================================================#
import re
import sys
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from PIL import Image




CKPT_PATH_G = '../input/nih-chest-xray-224-exp/AG_CNN_Global_epoch_1.pkl' 
CKPT_PATH_L = '../input/nih-chest-xray-224-exp/AG_CNN_Local_epoch_2.pkl' 
CKPT_PATH_F = '../input/nih-chest-xray-224-exp/AG_CNN_Fusion_epoch_23.pkl'

save_model_path = '../output/'
save_model_name = 'AG_CNN'

LR_G = 1e-8
LR_L = 1e-8
LR_F = 1e-3
num_epochs = 50
BATCH_SIZE = 32

classes = ('positive', 'negetive')

N_CLASSES = 1

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 

# 超参数设置
EPOCH = 20   #遍历数据集次数
pre_epoch = 0
BATCH_SIZE = 8      #训练批处理尺寸(batch_size)
LR = 0.01        #学习率
validation_split = .2
shuffle_dataset = True
random_seed= 42





def main():
    print('********************load data********************')
    # 准备数据集并预处理
    transforms_norm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=transforms_norm)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    print('********************load data succeed!********************')


    print('********************load model********************')
    # initialize and load the model
    Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        # to load state
        # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
        state_dict = checkpoint['state_dict']
        remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            ori_key =  key
            key = key.replace('densenet121.','')
            #print('key',key)
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            #print('new_key',new_key)
            if '.0.' in new_key:
                new_key = new_key.replace('0.','')
            state_dict[new_key] = state_dict[ori_key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[ori_key]
        
        Global_Branch_model.load_state_dict(state_dict)
        Local_Branch_model.load_state_dict(state_dict)
        print("=> loaded baseline checkpoint")
        
    else:
        print("=> no checkpoint found")

    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    criterion = nn.BCELoss()
    optimizer_global = optim.Adam(Global_Branch_model.parameters(), lr=LR_G, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_global = lr_scheduler.StepLR(optimizer_global , step_size = 10, gamma = 1)
    
    optimizer_local = optim.Adam(Local_Branch_model.parameters(), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_local = lr_scheduler.StepLR(optimizer_local , step_size = 10, gamma = 1)
    
    optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion , step_size = 15, gamma = 0.1)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch , num_epochs - 1))
        print('-' * 10)
        #set the mode of model
        lr_scheduler_global.step()  #about lr and gamma
        lr_scheduler_local.step() 
        lr_scheduler_fusion.step() 
        Global_Branch_model.train()  #set model to training mode
        Local_Branch_model.train()
        Fusion_Branch_model.train()

        running_loss = 0.0
        #Iterate over data
        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            #
            target_var = target_var.unsqueeze(1).float()

            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            optimizer_fusion.zero_grad()

            # compute output
            output_global, fm_global, pool_global = Global_Branch_model(input_var)
            
            patchs_var = Attention_gen_patchs(input,fm_global)

            output_local, _, pool_local = Local_Branch_model(patchs_var)
            #print(fusion_var.shape)
            output_fusion = Fusion_Branch_model(pool_global, pool_local)
            #

            # loss
            loss1 = criterion(output_global, target_var)
            loss2 = criterion(output_local, target_var)
            loss3 = criterion(output_fusion, target_var)
            #

            loss = loss1*0.8 + loss2*0.1 + loss3*0.1 

            if (i%10) == 0: 
                print('step: {} totalloss: {loss:.3f} loss1: {loss1:.3f} loss2: {loss2:.3f} loss3: {loss3:.3f}'.format(i, loss = loss, loss1 = loss1, loss2 = loss2, loss3 = loss3))

            loss.backward() 
            optimizer_global.step()  
            optimizer_local.step()
            optimizer_fusion.step()

            #print(loss.data.item())
            running_loss += loss.data.item()
            #break
            '''
            if i == 40:
                print('break')
                break
            '''

        epoch_loss = float(running_loss) / float(i)
        print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))

        print('*******testing!*********')
        test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,test_loader)
        #break

        #save
        if epoch % 1 == 0:
            save_path = save_model_path
            torch.save(Global_Branch_model.state_dict(), save_path+save_model_name+'_Global'+'_epoch_'+str(epoch)+'.pkl')
            print('Global_Branch_model already save!')
            torch.save(Local_Branch_model.state_dict(), save_path+save_model_name+'_Local'+'_epoch_'+str(epoch)+'.pkl')
            print('Local_Branch_model already save!')
            torch.save(Fusion_Branch_model.state_dict(), save_path+save_model_name+'_Fusion'+'_epoch_'+str(epoch)+'.pkl')            
            print('Fusion_Branch_model already save!')

        time_elapsed = time.time() - since
        print('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    

def test(model_global, model_local, model_fusion, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            if i % 2000 == 0:
                print('testing process:',i)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)

            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
            
    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROC_avg = np.array(AUROCs_g).mean()
    print('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROC_avg = np.array(AUROCs_l).mean()
    print('\n')
    print('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

    AUROCs_f = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROCs_f).mean()
    print('\n')
    print('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

if __name__ == '__main__':
    main()




