import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
 
#step1 load dataset 
iris = load_iris() 
x=iris.data 
y=iris.target 
 
#step2 pre proceswsing-scalling the data 
scaler  = StandardScaler() 
x_scaled = scaler.fit_transform(x) 
 
kmeans = KMeans(n_clusters=3) 
kmeans.fit(x_scaled) 
 
#step4 get  cluster center and labels 
centroid = kmeans.cluster_centers_ 
labels=kmeans.labels_ 
 
#step5 visualize the results 
plt.figure(figsize=(8,6)) 
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=labels,cmap='viridis') 
 
#plot the centroids  
plt.scatter(centroid[:,0],centroid[:,1],c='red',s=300,marker='X',label="Centroids") 
 
#Adding the labels 
plt.title('kMeans clustering algorithm ') 
plt.xlabel('Scaled sepal length') 
plt.ylabel('Scaled sepal width') 
plt.legend() 
plt.show()
