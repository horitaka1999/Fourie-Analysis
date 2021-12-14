from math import sqrt
import cv2
import numpy as np
from matplotlib import pyplot as plt
class ContorProduce:#imageは二つ画像(0,1)
    def __init__(self,image) -> None:
        self.image = np.array(image,dtype = np.uint8)
        self.image = np.flipud(self.image)
        self.image *= 255
        self.contours,self.hierarchy = cv2.findContours(self.image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    def ContorNumber(self):
        return len(self.contours)
        
    def produce(self,index):#[[x,y] ...]
        self.rev = []
        self.cnt = self.contours[index]
        for tmp in self.contours[index]:
            for x,y in tmp:
                self.rev.append((x,y))
        return np.array(self.rev)

    def culcArclength(self):
        length = cv2.arcLength(self.cnt,True)
        return length

    def convex_hull(self,index):
        hull = cv2.convexHull(self.contours[index],returnPoints =False)
        rev = []
        for i in range(len(hull)):
            rev.append(hull[i][0])
        return rev

    
'''
image = np.load('./data/sliced.npy')
cp = ContorProduce(image)
cp.viewCopntors(0)
print(cp.produce(0))

'''


