# Freddy @JinZhouWangBa, Tongji, Shanghai, China
# Jan 3, 2018

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import time
import os
import math
import random
import PIL.Image
import cv2
from termcolor import colored

#from utils import *


''' Hyper pamrams '''
#dtype=double
learning_rate=1e-3

def auto_train(model, loss_fn, optimizer):
    epoch_num = 1
    for epoch in range(epoch_num):
        count=0
        files = os.listdir('./data/digits')
        shuffled = sorted(files, key=lambda L: random.random())
        for img in shuffled:
            if('png' not in img): continue
            digit_img = PIL.Image.open('./data/digits/'+img)
            names = img.split('-')
            y_var = int(names[0])
            count+=1
            model.train()
            optimizer.zero_grad()
            # x_var: digit_img, y_var: label
            pil_img = np.array(digit_img.resize((32, 32), PIL.Image.NEAREST))
            pil_img = np.expand_dims(pil_img, axis=0)
            transform = transforms.Compose([transforms.ToTensor()])
            x_var = transform(pil_img).resize_(1,1,32,32)
            x_var = Variable(x_var)
            loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)
            scores = None
            for img in loader:
                scores = model(x_var)
            scores_cp = (scores.data).cpu().numpy()
            pred = scores_cp.argmax()
            if(y_var>=10 and y_var<=99):
                y_var = 10 # label 10: 2-digit number
            elif(y_var>=100 and y_var<=999):
                y_var = 11 # label 11: 3-digit number
            label = Variable(torch.from_numpy(np.array([y_var])))
            loss = loss_fn(scores, label)
            right = colored('XXX','red')
            if(y_var == pred): right = colored('VVV','green')
            print('epoch: '+str(epoch+1)+', iter '+str(count)+', loss: '+str(to_np(loss)[0])+'; Prediction: '+str(pred)+', Real value: '+str(y_var)+ '  ', right)
            loss.backward()
            optimizer.step()

def train(model, loss_fn, optimizer):
    count=0
    while(True):
        count+=1
        model.train()
        optimizer.zero_grad()
        # x_var: digit_img, y_var: label
        digit_img=None
        y_var=int(input('Press corresponding label value to take the screenshot, enter -1 to stop: '))
        if(y_var!="-1"):
            digit_img = clipper()
        else:
            break
        #pil_img = np.array(digit_img.resize((32, 32), PIL.Image.NEAREST)).reshape((1,4,32,32))
        pil_img = np.array(digit_img.resize((32, 32), PIL.Image.NEAREST))
        #pil_img.show()
        #print(pil_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        #x_var = torch.from_numpy(pil_img)
        x_var = transform(pil_img).resize_(1,4,32,32)
        x_var = Variable(x_var)
        loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)
        scores = None
        for img in loader:
            scores = model(x_var)
        scores_cp = (scores.data).cpu().numpy()
        preds = scores_cp.argmax()
        #print('Prediction: '+str(preds))
        if(y_var == -1): break
        if(y_var>=10 and y_var<=99):
            y_var = 10 # label 10: 2-digit number
        elif(y_var>=100 and y_var<=999):
            y_var = 11 # label 11: 3-digit number
        
        right = colored('XXX','red')
        if(y_var == preds): right = colored('VVV','green')
        print('Pred: '+ str(preds) + ', Real Value: '+ str(y_var) + ' ',right)      

        # store the sample for later training
        digit_img.save('./data/digits/'+str(y_var)+'-'+str(random.random())+'.png')
        
        label = Variable(torch.from_numpy(np.array([y_var])))
        #print(label.size())
        loss = loss_fn(scores, label)
        print('iter '+str(count)+', loss: '+str(to_np(loss)[0]))
        loss.backward()
        optimizer.step()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size = 3, padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                nn.Conv2d(32,32,kernel_size = 3, padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size = 3, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,64,kernel_size = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size = 1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.fc3 = nn.Linear(8*8*128,12)
        #self.pl = nn.AvgPool2d((,12))
        self.sm = nn.Softmax()

    def forward(self, x):
        #print(x.size())
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        out = self.layer5(out)
        #print(out.size())
        out = out.view(out.size(0),-1)
        #print(out.size())
        out = self.fc3(out)
        return self.sm(out)


if __name__ == '__main__':
    print('create DNN...')
    #cnn = CNN().type(dtype)
    cnn = CNN()
    if(torch.cuda.is_available()):
        cnn.cuda()
        
    if(int(input('Enter \'1\' to load old model, others not to: '))==1):
        cnn = torch.load("digit_recognizer.pkl")
        
        #loss_fn = nn.CrossEntropyLoss().type(dtype)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(cnn.parameters(), lr=learning_rate, lr_decay=0.0, weight_decay=0)
    
    print('start training...')
    if(input('Press \'enter\' to auto train, else not to: ')==''):
        auto_train(cnn, loss_fn, optimizer)
    #train(cnn, loss_fn, optimizer)
    
    if(int(input('Enter \'1\' to save the model, others to quit: '))==1):
        torch.save(cnn, "digit_recognizer.pkl")
