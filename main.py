from apex import amp
import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
#from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from  albumentations  import (
        HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
        IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Resize

    )
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dir_csv = './annos'
dir_train_img = './carpet/'    #'./train'
dir_test_img = './carpet/'

n_classes = 8
n_epochs = 20
batch_size = 1

# Functions
class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.path, self.data.loc[idx, 'ID'])
        img = cv2.imread(img_name)   
        
        if self.transform:       
            
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['bent','color','contamination','cut','good','hole', 'scratch','thread']])
            return {'image': img, 'labels': labels}    
        
        else:      
            
            return {'image': img}
			

# Data loaders
transform_train = Compose([
    CLAHE(),
    RandomRotate90(),
    Transpose(),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    Blur(blur_limit=3),
    OpticalDistortion(),
    GridDistortion(),
    HueSaturationValue(),
    Resize(224, 224, always_apply=False, interpolation=1),
    ToTensor(),
])

transform_test= Compose([
    Resize(224, 224, always_apply=False, interpolation=1),
    ToTensor(),
])

train_dataset = IntracranialDataset(
    csv_file='./annos/train.csv', path=dir_train_img, transform=transform_train, labels=True)

test_dataset = IntracranialDataset(
    csv_file='./annos/val.csv', path=dir_test_img, transform=transform_test, labels=True)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

'''
#from apex.parallel import DistributedDataParallel
# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
#model = torch.load('./output/best/epoch_3.pth')
model.fc = torch.nn.Linear(2048, n_classes)
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model,device_ids=device_ids)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 5e-5}]
optimizer = optim.Adam(plist, lr=5e-5)
#optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
#optimizer.module.step()

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#model = DistributedDataParallel(model)
# Train

for epoch in range(n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()    
    tr_loss = 0
    
    tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(tk0):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if step % 20 == 0 and step != 0:
            st_loss = tr_loss/step
            print("step:{}/{},loss:{:4f}".format(step,len(data_loader_train),st_loss))


    torch.save(model,'./output/epoch_{}.pth'.format(epoch))
    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))
'''

# Inference

device = torch.device("cuda:0")
model = torch.load('./output/epoch_12.pth')
for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset) * n_classes, 1))

for i, x_batch in enumerate(tqdm(data_loader_test)):
    x_input = x_batch["image"]
    target = x_batch['labels']
    x_input = x_input.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    
    with torch.no_grad():
        acc = 0
        pred = model(x_input)
        #pred = pred.x_batch.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print(pred.max(1,keepdim=False)[1],target.max(1,keepdim=False)[1])
        if not torch.equal(pred.max(1,keepdim=False)[1].view(1,-1),target.max(1,keepdim=False)[1].view(1,-1)):
            print("Error")
            #acc = acc+1
          
#print(correct,len(data_loader_test))			
