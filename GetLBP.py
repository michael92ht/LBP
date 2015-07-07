# -*- coding: utf-8 -*-

import numpy as np
import cv2
from PIL import Image
from pylab import*

def calc_sum(r):
    res_sum = 0
    while r!=0:
        res_sum = res_sum + r % 2
        r /= 2
    return res_sum  

def get_min(arr):
    values=[]
    circle=arr
    circle.extend(arr)
    for i in range(0,9):
        j=0
        sum=0
        bit_num=1
        while j<8:
            sum+=circle[i+j]*bit_num
            bit_num*=2
            j+=1
        values.append(sum)
    return min(values)
    
    
        
class LBP:
    def __init__(self, cell):
        self.cell=cell
    def describe(self,image):
        image_array=np.array(Image.open(image).convert('L'))
        return image_array
    def lbp_basic(self,image_array):
        new_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute(image_array,i,j)
                bit_num=1
                result=0
                for s in sum:
                    result+=s*bit_num
                    bit_num=bit_num*2
                new_array[i,j]=result
        return new_array   
    def calute(self,image_array,i,j):
        sum=[]
        if image_array[i-1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)

        return sum
    def show_hist(self,img):
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        #hist = cv2.normalize(hist).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim([0,256])
        plt.show()
        
    def show_advance_hist(self,img):
        hist = cv2.calcHist([img],[0],None,[36],[0,36])
        #hist = cv2.normalize(hist).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim([0,36])
        plt.show()
    def show_uniform_hist(self,img):
        hist = cv2.calcHist([img],[0],None,[59],[0,59])
        #hist = cv2.normalize(hist).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim([0,59])
        plt.show()
    def show_uniform_advance_hist(self,img):
        hist = cv2.calcHist([img],[0],None,[10],[0,10])
        #hist = cv2.normalize(hist).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim([0,10])
        plt.show()
    def show(self,image_array):
        cv2.imshow('dst',image_array)
        cv2.waitKey(0)
        #cv2.destroyAllWindow()    
    
    def lbp_uniform(self,image_array):
        new_array=np.zeros(image_array.shape, np.uint8)
        basic=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic[i,j]^k
                 num=calc_sum(xor)
                 if num<=2:
                     new_array[i,j]=basic[i,j]
                 else:
                     new_array[i,j]=9
        return new_array
    
    def lbp_uniform_advance(self,image_array):
        new_array=np.zeros(image_array.shape, np.uint8)
        basic=self.lbp_advance(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic[i,j]^k
                 num=calc_sum(xor)
                 if num<=2:
                     new_array[i,j]=basic[i,j]
                 else:
                     new_array[i,j]=9
        return new_array      
    def lbp_advance(self,image_array):
        new_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute(image_array,i,j)
                new_array[i,j]=get_min(sum)
        return new_array   


lbp=LBP(0)
img=lbp.describe(r"d:\cv\000.jpg")
adun=lbp.lbp_uniform(img)
lbp.show_uniform_hist(adun)
lbp.show(adun)
#arr=lbp.lbp_basic(img)
#ad=lbp.lbp_advance(img)
#lbp.show_advance_hist(ad)
#un=lbp.lbp_uniform(img)
#lbp.show_uniform_hist(un)
#lbp.show(un)
#cv2.imshow('src',img)
#cv2.waitKey(0)
print 'done!'



