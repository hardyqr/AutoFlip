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


def drawer(bbox, gray2):
    # draw 
    for box in bbox:
        [x,y,w,h] = box
        cv2.rectangle(gray2,(x,y),(x+w,y+h),(250,250,250),2)
    toimage(gray2).show()
    return 0

def get_bbox(contours,gray2):
    bbox = []
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000: continue
        #mask = np.zeros(gray2.shape,np.uint8)
        #cv2.drawContours(mask,[cnt],0,255,-1)
        [x,y,w,h] = cv2.boundingRect(cnt)
        #cv2.imshow('Features', gray2)
        keep = [x,y,w,h]
        to_keep = True
        for kept in bbox:
            if(abs(kept[0]-keep[0])<10):to_keep=False # duplicate bbox
        if(h<150): to_keep=False
        if(to_keep):
            count+=1
            bbox.append(keep)
            print('bbox got: ', keep)

            #drawer(bbox,gray2)
            label = int(input('enter label: '))
            cv2.imwrite('./data/digits/'+str(label)+'-'+str(random.random())+'.png', gray2[y:y+h,x:x+w])

    cv2.destroyAllWindows()
    return bbox

def segment_digitGray_and_gameRGB():
    im = ImageGrab.grab(bbox=(115, 50, 675, 999999))  # X1,Y1,X2,Y2
    image_np = np.array(im)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #thresh = cv2.adaptiveThreshold(blur,254,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret,thresh = cv2.threshold(blur,250,250,cv2.THRESH_BINARY)
    #gray = cv2.Canny(image_np, 20, 80)
    #toimage(gray).show()
    
    digits_gray = thresh[0:int(gray.shape[0]/6),0:gray.shape[1]]
    #toimage(digits_gray).show()

    game_rgb = image_np[int(gray.shape[0]/6):gray.shape[0],0:gray.shape[1]]
    #toimage(game_rgb).show()

    gray2,contours,hierarchy = cv2.findContours(digits_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.destroyAllWindows()
    return gray2,contours,game_rgb


def gen_train():
    while(True):
        print('new round')
        gray2,contours,game_rgb = segment_digitGray_and_gameRGB()
        _=get_bbox(contours,gray2)
        cv2.destroyAllWindows()
        print('finish')

while(True):
    print('new round')
    gray2,contours,game_rgb = segment_digitGray_and_gameRGB()
    _=get_bbox(contours,gray2)
    cv2.destroyAllWindows()
    print('finish')


'''
if __name__ == "__main__":

    #gray2,contours,game_rgb = segment_digitGray_and_gameRGB()
    #toimage(game_rgb).show()
    #bbox = get_bbox(contours,gray2)
    #drawer(bbox,gray2)

    gen_train()

'''
