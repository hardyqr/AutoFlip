# Freddy @JinZhouWangBa, Tongji, Shanghai, China
# Jan 3, 2018

import time
import PIL.Image
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from pymouse import PyMouse
from pykeyboard import PyKeyboard
import pyscreenshot as ImageGrab
import numpy as np
from termcolor import colored
import math
from scipy.misc import toimage


from digit_recognizer import CNN
from digit_detector import *
from RL import fake_PG

''' digit recognizer to obtain reward '''
digit_cnn = torch.load("digit_recognizer.pkl")

if(torch.cuda.is_available()):
    digit_cnn.cuda()


''' Policy Gradient Network training '''

model = fake_PG()
if(torch.cuda.is_available()):
    model.cuda()


if __name__ == '__main__':
    
    gray2,contours,game_rgb = segment_digitGray_and_gameRGB(get_screenshot())
    prev_score=0
    while(True):
        #print(game_rgb.shape)
        toimage(game_rgb).show()
        game_rgb = PIL.Image.fromarray(game_rgb)
        game_rgb = np.array(game_rgb.resize((128, 128), PIL.Image.NEAREST))
        toimage(game_rgb).show()
        #print(pil_img.shape)
        #game_rgb = np.expand_dims(game_rgb, axis=0)
        transform = transforms.Compose([transforms.ToTensor()])
        #x_var = torch.from_numpy(pil_img)
        #game_rgb = np.transpose(game_rgb,(0,3,1,2))
        print(game_rgb.shape)
        game_var = Variable(transform(game_rgb).resize_(1,3,128,128))
        #loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)

        pred=model(game_var)
        pred_cp = (pred.data).cpu().numpy()
        t=pred_cp.argmax()*100
        os.system('adb shell input swipe 100 100 100 100 '+str(t+100))
        t = 2.0
        time.sleep(t)
        
        gray2,contours,game_rgb = segment_digitGray_and_gameRGB(get_screenshot())
        bbox,digits=get_bbox(contours,gray2)
        score=-1
        for i in range(len(bbox)):
            if(i==0):score=0
            im = PIL.Image.fromarray(digits[len(bbox)-1-i])
            pil_img = np.array(im.resize((32, 32), PIL.Image.NEAREST))
            #pil_img.show()
            #print(pil_img.shape)
            pil_img = np.expand_dims(pil_img, axis=0)
            transform = transforms.Compose([transforms.ToTensor()])
            #x_var = torch.from_numpy(pil_img)
            x_var = Variable(transform(pil_img).resize_(1,1,32,32))
            #loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)
            scores = digit_cnn(x_var)
            scores_cp = (scores.data).cpu().numpy()
            #pred = colored(str(scores_cp.argmax()), 'red')
            #print('current digit: '+pred)
            score+=int(scores_cp.argmax())*pow(10,i)
            delta = score - prev_score
            prev_score = score

        if(score==-1): # fail
            print(colored('You lost.','red'))

            # restart
            os.system('adb shell input swipe 400 1700 400 1700 '+str(10))
            prev_score=0
            
            transform = transforms.Compose([transforms.ToTensor()])
            loss = Variable(transform(np.array(10.0)))


        else: # game continues
            pred = colored(str(score), 'red')
            print('current score: '+pred)
            #delta = score - prev_score
            print('delta(score): '+str(delta))

            transform = transforms.Compose([transforms.ToTensor()])
            if(delta==0):delta=0.2
            loss=Variable(transform(np.array(float(1/delta-1/32))))
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss.backward()
        optimizer.step()
        input('Press \'enter\' to continue...')

