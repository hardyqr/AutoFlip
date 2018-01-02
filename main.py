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

from data.digits.generater import clipper
from digit_recognizer import CNN
  

''' digit recognizer to obtain reward '''
digit_cnn = torch.load("digit_recognizer.pkl")
if(torch.cuda.is_available()):
    digit_cnn.cuda()

m = PyMouse()
k = PyKeyboard()

x_dim, y_dim = m.screen_size()
#m.click(x_dim/2, y_dim/2, 1)

while(True):
    im=ImageGrab.grab(bbox=(0,0,x_dim/2,y_dim))
    digit_img=clipper()
    pil_img = np.array(digit_img.resize((32, 32), PIL.Image.NEAREST))
    #pil_img.show()
    #print(pil_img.shape)
    transform = transforms.Compose([transforms.ToTensor()])
    #x_var = torch.from_numpy(pil_img)
    x_var = transform(pil_img).resize_(1,4,32,32)
    x_var = Variable(x_var)
    loader = torch.utils.data.DataLoader(dataset=pil_img, batch_size=1)
    scores = digit_cnn(x_var)
    scores_cp = (scores.data).cpu().numpy()
    pred = colored(str(scores_cp.argmax()), 'red')

    print('current score: '+pred)
    input('Press \'enter\' to continue...')
    #m.press(x_dim/3, y_dim/3)
    
    t = 0.4
    #time.sleep(t)
    #m.release(x_dim/3, y_dim/3)

