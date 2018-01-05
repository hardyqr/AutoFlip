# Freddy @MaanCoffee, Zijin, Chengdu, China
# Jan 4, 2018

import sys
import numpy as np
import cv2
#from matplotlib import pyplot as plt
import PIL.Image
from scipy.misc import toimage
import pyscreenshot as ImageGrab
import random
import os
from operator import itemgetter


def drawer(bbox, gray2):
    # draw 
    for box in bbox:
        [x,y,w,h] = box
        cv2.rectangle(gray2,(x,y),(x+w,y+h),(1,255,1),2)
    toimage(gray2).show()
    return 0

def get_bbox(contours,gray2):
    bbox = []
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000: continue # too small, not number
        #mask = np.zeros(gray2.shape,np.uint8)
        #cv2.drawContours(mask,[cnt],0,255,-1)
        [x,y,w,h] = cv2.boundingRect(cnt)
        #cv2.imshow('Features', gray2)
        keep = [x,y,w,h]
        #print(keep)
        to_keep = True
        for kept in bbox:
            if(abs(kept[0]-keep[0])<5):to_keep=False # duplicate bbox
        if(h>100 or h<50): to_keep=False
        if(to_keep):
            count+=1
            bbox.append(keep)
            print('bbox got: ', keep)
            
            # save training samples, uncoment when using it
            #gray2_copy=gray2.copy()
            #drawer(bbox,gray2_copy)
            #label = int(input('enter label: '))
            #cv2.imwrite('./data/digits/'+str(label)+'-'+str(random.random())+'.png', gray2[y:y+h,x:x+w])

    cv2.destroyAllWindows()
    bbox = sorted(bbox, key=itemgetter(0))# sort by 'x'
    digits=[]
    for i in range(len(bbox)):
        [x,y,w,h]=bbox[i]
        digits.append(gray2[y:y+h,x:x+w]) # digtis: LSD on the left
    return bbox, digits

def get_screenshot():
    os.system('adb shell screencap -p /sdcard/state.png')
    os.system('adb pull /sdcard/state.png ./data/state.png')
    im = cv2.imread('./data/state.png')
    return im



def segment_digitGray_and_gameRGB(im):
    image_np = np.array(im)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    #ret,thresh = cv2.threshold(blur,250,1,cv2.THRESH_BINARY)
    ret,thresh = cv2.threshold(blur,128,255,cv2.THRESH_BINARY)
    #gray = cv2.Canny(image_np, 20, 80)
    #toimage(gray).show()
    
    digits_gray = thresh[0:int(gray.shape[0]/6),0:gray.shape[1]]
    kernel = np.ones((4,4),np.uint8)
    digits_gray = cv2.erode(digits_gray,kernel,iterations = 2)
    #toimage(digits_gray).show()

    game_rgb = image_np[int(gray.shape[0]/6):gray.shape[0],0:gray.shape[1]]
    #toimage(game_rgb).show()

    gray2,contours,hierarchy = cv2.findContours(digits_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.destroyAllWindows()
    return gray2,contours,game_rgb


def gen_train():
    while(True):
        print('new round')
        im = get_screenshot()
        gray2,contours,game_rgb = segment_digitGray_and_gameRGB(im)
        _,_=get_bbox(contours,gray2)
        cv2.destroyAllWindows()
        print('finish')




if(__name__ == '__main__'):
    gen_train()


