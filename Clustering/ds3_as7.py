#!/usr/bin/env python
# coding: utf-8

# In[16]:


#1
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("iris.csv") #original dataframe
X=np.array([df["SepalLengthCm"],df["SepalWidthCm"],df["PetalLengthCm"],df["PetalWidthCm"]]).T #original numpy array
pca = PCA(n_components=2)
pca.fit(Scaled_data)
X1=pca.transform(X) #tranformed numpy array
df1=pd.DataFrame(X1) #transformed dataframe
plt.scatter(df1[0],df1[1])
plt.title("reduced 2d data")
plt.ylabel("component 2")
plt.xlabel("component 1")
plt.show()
pca1=PCA(n_components=4) #pca with 4 components
pca1.fit(Scaled_data)
plt.bar(['1','2','3','4'],pca1.explained_variance_)
plt.title("Eigenvalue vs components")
plt.ylabel("Eigenvalue")
plt.xlabel("components")


# In[17]:


#2
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
from sklearn.cluster import KMeans
K = 3
kmeans = KMeans(n_clusters=K,random_state=42)
kmeans.fit(np.array([df1[0],df1[1]]).T)
kmeans_prediction = kmeans.predict(np.array([df1[0],df1[1]]).T)
plt.scatter(df1[0], df1[1], c=kmeans_prediction, s=20, cmap='summer')
scatter=plt.scatter(df1[0], df1[1], c=kmeans_prediction, s=20, cmap='summer')
plt.legend(handles=scatter.legend_elements()[0], labels=["iris setosa","iris versicolor","iris virginica"],title="species")
centers = kmeans.cluster_centers_
scatter=plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.title("K-means (K=3) clustering on Iris flower dataset")
plt.show()
print("distortion measure::",kmeans.inertia_) #distortion measure
from sklearn.metrics import adjusted_rand_score
true_labels=[]
for i in df["Species"]:
    if(i=="Iris-setosa"):
        true_labels.append(0)
    elif(i=="Iris-virginica"):
        true_labels.append(2)
    else:
        true_labels.append(1)
ari_kmeans = purity_score(true_labels, kmeans.labels_)
print("purity score::",ari_kmeans) #purity score


# In[22]:


#3
l=[2,3,4,5,6,7]
dist,pur_sc=[],[]
for i in l:
    K = i
    kmeans = KMeans(n_clusters=K,random_state=42)
    kmeans.fit(np.array([df1[0],df1[1]]).T)
    kmeans_prediction = kmeans.predict(np.array([df1[0],df1[1]]).T)
    dist.append(kmeans.inertia_)
    pur_sc.append(purity_score(true_labels, kmeans.labels_))
#elbow method
plt.plot(l,dist)
plt.title(" Number of clusters(K) vs. distortion measure")
plt.xlabel("Number of clusters")
plt.ylabel("distortion measure")
pd.DataFrame(np.array([l,pur_sc]).T,columns=["no. of clusters","purity score"])


# In[23]:


#4
from sklearn.mixture import GaussianMixture as GMM
import scipy.stats
K = 3
gmm = GMM(n_components = K,random_state=42)
gmm.fit(np.array([df1[0],df1[1]]).T)
GMM_prediction = gmm.predict(np.array([df1[0],df1[1]]).T)
scatter=plt.scatter(df1[0], df1[1], c=GMM_prediction, s=20, cmap='summer')
plt.legend(handles=scatter.legend_elements()[0], labels=["iris setosa","iris versicolor","iris virginica"],title="species")
centers = kmeans.cluster_centers_
centers = np.empty(shape=(gmm.n_components, np.array([df1[0],df1[1]]).T.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(np.array([df1[0],df1[1]]).T)
    centers[i, :] = np.array([df1[0],df1[1]]).T[np.argmax(density)]
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.title("GMM (K=3) clustering on Iris flower dataset")
plt.show()
print("distortion measure",gmm.score(np.array([df1[0],df1[1]]).T)*len(np.array([df1[0],df1[1]]).T))
ari_kmeans = purity_score(true_labels, GMM_prediction)
print("purity score",ari_kmeans) #purity score


# In[24]:


#5
dist=[]
pur_sc=[]
for i in l:
    K = i
    gmm = GMM(n_components = K,random_state=42)
    gmm.fit(np.array([df1[0],df1[1]]).T)
    GMM_prediction = gmm.predict(np.array([df1[0],df1[1]]).T)
    dist.append(gmm.score(np.array([df1[0],df1[1]]).T)*len(np.array([df1[0],df1[1]]).T))
    pur_sc.append(purity_score(true_labels, GMM_prediction))
plt.plot(l,dist)
plt.title(" Number of clusters(K) vs. distortion measure")
plt.xlabel("Number of clusters")
plt.ylabel("distortion measure")
pd.DataFrame(np.array([l,pur_sc]).T,columns=["no. of clusters","purity score"])


# In[21]:


#6
from sklearn.cluster import DBSCAN
import numpy as np
import scipy as sp
from scipy import spatial as spatial
def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
            # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new
eps=[1,5]
min_samples=[4,10]
pur_sc=[]
for i in eps:
    for j in min_samples:
        dbscan_model=DBSCAN(eps=i, min_samples=j).fit(np.array([df1[0],df1[1]]).T)
        #DBSCAN_predictions = dbscan_model.labels_
        DBSCAN_predictions=dbscan_predict(dbscan_model,np.array([df1[0],df1[1]]).T)
        scatter=plt.scatter(df1[0], df1[1], c=DBSCAN_predictions, s=20, cmap='summer')
        plt.legend(handles=scatter.legend_elements()[0], labels=["iris setosa","iris versicolor","iris virginica"],title="species")
        plt.title("DBSCAN clustering")
        plt.show()
        pur_sc.append([i,j,(purity_score(true_labels, DBSCAN_predictions))])
pd.DataFrame(pur_sc,columns=["eps","minimum_samples","purity score"])


# In[ ]:




