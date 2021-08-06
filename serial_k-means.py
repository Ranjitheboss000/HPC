import math
import csv
import time
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


#===================================================Euclidian distance=====================
def eucl_distance(point_one, point_two):                  #define function to measure Euclidian distance between point_one and point_two
	if(len(point_one) != len(point_two)):                 #check for an error (len(1st point)is not equal len(2nd point)) 
		raise Exception("Error: non comparable points")   #check for an error 

	sum_diff = 0.0                                                 #set variable sum_diff
	for i in range(len(point_one)):                                #iterating take difference between each elem from 1st and 2nd point
		diff = pow((float(point_one[i]) - float(point_two[i])), 2) #and square this difference. 
		sum_diff += diff                                           #As the result summing all these differencess and equating to sum_diff
	final = math.sqrt(sum_diff)                                    #sqrt from sum_diff will be Euqlidian dist betw(1st and 2nd points)
	return final
#===================================================Devide data set to further scattering=====================

global dimensions, num_clusters, num_points,dimensions,data,flag

#===============================================reading and preparing data set======================
print("Enter the number of clusters you want to make: ")
num_clusters = int(input())
start_time = time.time()
with open('3D_spatial_network.csv','r') as f:
    reader = csv.reader(f)
    data = list(reader)

data.pop(0)
for i in range (len(data)):
    data[i].pop(0)
data=np.array(data).astype(np.float)
#print("Hey",len(data))
data=data[0:10000]
#Print(data)
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data).labels_
#====================================================================================================
start_time = time.time() 										#turn on a timer which allows us to estimate performane of this algorithm

#================================================Initialize centroids matrix=========================
initial=[]
for i in range(num_clusters):
    initial.append(data[i])
initial=np.vstack(initial)
#====================================================================================================

num_points = len(data)                                    #number of rows
dimensions = len(data[0])                                 #number of columns

#====================================================================================================

flag= True
while flag:

	cluster_vector=[]
	dist =np.zeros((len(data),len(initial)))

	for j in range(len(initial)):
		for i in range(len(data)):
			dist[i][j]=np.linalg.norm(initial[j]-data[i])

	#===================================Initilize lable for each sample in each process======================
	for i in range (len(dist)):										#iterable take each raw in dist matrix and 
		cluster_vector.append(np.argmin(dist[i])+1)                 #find column index of min value (this index is number of centroid)

	#===================================Calculating the number of samples in each cluster_vector====================
	Q_clusts=collections.Counter(cluster_vector)							
	
	centroid=np.zeros((len(initial),len(initial[0])))
	for k in range (1,num_clusters+1):
		indices = [i for i, j in enumerate(cluster_vector) if j == k]
		centroid[k-1]=np.divide((np.sum([data[i] for i in indices], axis=0)).astype(np.float),Q_clusts[k])

	if np.all(centroid==initial):
		flag=False
	else:
		initial= centroid 

print ("Execution time %s seconds" % (time.time() - start_time))
print('adjusted_rand_score',adjusted_rand_score(kmeans,cluster_vector))

	

