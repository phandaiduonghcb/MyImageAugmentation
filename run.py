from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import pickle as pkl
import os
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--quantity',default=5,type=int)
opt = parser.parse_args()
directory = opt.dir
n = opt.quantity
#directory = 'TEST'
#n = 5

#Them random crop
l = [f for f in listdir(directory) if isfile(join(directory,f)) and f[-3:]=='jpg']
first = True
temp = ''
for name in l:
    print(directory+name)
    img = cv2.imread(directory+name)[:,:,::-1] #OpenCV uses BGR channels
    t = img.shape
    height,width = t[0],t[1]
    try:
        label_file = f = open(directory + name[:-3] + "txt")
    except:
        os.remove(directory + name)

    box_arr = []
    boxes = label_file.readlines()
    for box in boxes:
        l = list(map(float,box.split()))
        X1 = l[1]*width-l[3]*width/2
        Y1 = l[2]*height-l[4]*height/2
        X2 = l[1]*width+l[3]*width/2
        Y2 = l[2]*height+l[4]*height/2
        C = l[0]
        
        box_arr.append([X1,Y1,X2,Y2,C])
    box_arr  = np.array(box_arr)
    bboxes = box_arr
    if first:
        temp = bboxes.copy()
        first = False 
    for i in range(n):
        try:
            seq = Sequence([RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(25), RandomShear(),RandomHSV(None, 40, 40),RandomCrop(0.5,(10,30),(10,30))],0.5)
            img_, bboxes_ = seq(img.copy(), bboxes.copy())
            plt.imsave(directory + name[:-4] + "_"+str(i)+".jpg",img_)
            f = open(directory + name[:-4]+"_"+str(i)+".txt",'w')
            for box in bboxes_:
                t = img_.shape
                height,width = t[0],t[1]
                C = box[4]
                p1 = (box[0] + box[2])/width/2
                p3 = (p1*width - box[0])/(width/2)
                p2 = (box[1] + box[3])/height/2
                p4 = (p2*height - box[1])/(height/2)
                f.write(str(int(C)) + ' '+ str(p1) + ' ' +str(p2) + ' ' + str(p3) + ' ' + str(p4)+ '\n')
            f.close()
        except:
            seq = Sequence([RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(25), RandomShear(),RandomHSV(None, 40, 40),RandomCrop(0.5,(10,30),(10,30))],0.5)
            img_, bboxes_ = seq(img.copy(), temp.copy())
            plt.imsave(directory + name[:-4] + "_"+str(i)+".jpg",img_)
            f = open(directory + name[:-4]+"_"+str(i)+".txt",'w')
            f.close()
            
