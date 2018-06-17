import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from preprocess import preprocessing
from sklearn.cluster import KMeans

# import image :
new_image = cv2.imread('./dataset/%d.png' %randint(0, 1000))

# preprocessing :
new_image = preprocessing(new_image)

# save orginal image :
cv2.imwrite("./org.jpg",new_image)


# k-means fit alg :
np_img = np.asarray(new_image)
np_img = np.argwhere(np_img == 255)
print(np_img[:,0])
kmeans = KMeans(n_clusters=5, random_state=0).fit(np_img)

# draw point on image :
new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
for  i in range(0,len(kmeans.cluster_centers_)):
    cv2.circle(new_image, ( int(kmeans.cluster_centers_[i][1]),int(kmeans.cluster_centers_[i][0])), 1, (0, 0, 255), 2)
cv2.imwrite("./kmeans.jpg", new_image)

# draw plot :
plt.plot( np_img[:,1],np_img[:,0], 'bo', label='Numbers',color='b')
plt.plot(  kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], 'bo', label='Cluster centers',color='r')





plt.title('K-means Clustering')
plt.legend()

plt.show()
