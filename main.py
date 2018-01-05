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

from digit_recognizer import CNN
from digit_detector import *

''' digit recognizer to obtain reward '''
digit_cnn = torch.load("digit_recognizer.pkl")
if(torch.cuda.is_available()):
    digit_cnn.cuda()

m = PyMouse()
k = PyKeyboard()

x_dim, y_dim = m.screen_size()
#m.click(x_dim/2, y_dim/2, 1)



if __name__ == '__main__':
    
    while(True):
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
            loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)
            scores = digit_cnn(x_var)
            scores_cp = (scores.data).cpu().numpy()
            #pred = colored(str(scores_cp.argmax()), 'red')
            #print('current digit: '+pred)
            prev_score = score
            score+=int(scores_cp.argmax())*pow(10,i)

        if(score==-1): # fail
            print(colored('You lost.'),'red')
        else: # game continues
            pred = colored(str(score), 'red')
            print('current score: '+pred)
            delta = score - prev_score
            print('delta(score): '+delta)

        input('Press \'enter\' to continue...')
    
    
    t = 0.4
    #time.sleep(t)
