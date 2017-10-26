# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:08:46 2017

@author: Administrator
"""
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from itertools import cycle

from sklearn.datasets import make_blobs



filePath = './UNIT352.csv'
tempData = pd.DataFrame(pd.read_csv(filePath))
L = tempData.size
print(L)
#
tempData = tempData[tempData['最终转角'] < 37]
#
tempData = tempData[tempData['最终转角'] > 32]


tempData = tempData[tempData['最终扭矩'] < 800]


#temp = np.transpose(np.array([tempData['最终扭矩'], tempData['最终转角']]))
#forAF =temp[0:1500] 
#np.savetxt('needss.csv',forAF, delimiter = ',')  



temp = np.transpose(np.array([tempData['最终扭矩'], tempData['最终转角']]))


forAF =temp[10000:20000] 



# print(forAF)

n_clusters_ =4    
KM = KMeans(n_clusters=n_clusters_, random_state=0).fit(forAF)     
KMLable = KM.labels_     
print("======lables======")
print(KMLable)
print("======clusters_center========")
KMcc = KM.cluster_centers_     
print(KMcc)
print("===========================================")


colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')    
for k, col in zip(range(n_clusters_), colors):
    class_members = KMLable == k    
    plt.plot(forAF[class_members, 0], forAF[class_members, 1], col + '.')   

plt.plot(KMcc[:, 0], KMcc[:, 1], '^', markerfacecolor=col,
         markeredgecolor='k', markersize=14)   

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#data,target=make_blobs(n_samples=10000,n_features=2,centers=[[108,35.5], [99.3,35.4], [103.5,35.4]])

#n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3], random_state =9

#在2D图中绘制样本，每个样本颜色不同
#yplot.scatter(data[:,0],data[:,1],c=target);
#pyplot.show()


#forAF =forAF[140000:150000] 
#np.savetxt('need.txt', forAF, delimiter = ',')  





