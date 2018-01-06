# Freddy @ChuanDaHuaYuan, Chengdu, China
# Jan 6, 2018

from termcolor import colored

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

