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
import os

from screen_reader.digit_detector import get_score_and_gameRGB
from screen_reader.digit_recognizer import CNN
from screen_reader.utils import *

if __name__ == '__main__':
    
    while(True):
        score, gameRGB = get_score_and_gameRGB()
        if(score==-1): # fail
            print(colored('You lost.','red'))
            # restart
            os.system('adb shell input swipe 400 1700 400 1700 '+str(10))
            
        else: # game continues
            pred = colored(str(score), 'red')
            print('current score: '+pred)
        input('Press \'enter\' to continue...')

