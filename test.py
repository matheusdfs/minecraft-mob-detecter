from sklearn.cluster import DBSCAN
import numpy as np


points = np.array([[ 721.35876 , 508.39603],
 [ 980.30853 , 583.47906],
 [ 985.2651  , 625.8798 ],
 [ 985.18787 , 607.4524 ],
 [ 987.29565 , 573.1455 ],
 [ 989.50867 , 563.41174],
 [1001.32355 , 586.4493 ],
 [1001.32355 , 586.4493 ],
 [1010.02954 , 606.9703 ],
 [ 865.3559  , 654.37976],
 [1016.058   , 580.4455 ],
 [1025.1371  , 561.8872 ],
 [1038.0002  , 586.44   ],
 [1042.5176  , 582.088  ],
 [1046.2219  , 626.6    ],
 [1047.137   , 614.63983],
 [1050.8229  , 601.9846 ],
 [ 645.1658  , 907.54   ]])

clustering = DBSCAN(eps=50, min_samples=3).fit(points)
np.where(clustering.labels_ == 0, True, False)

# print(clustering.labels_)

# print(points)

X = np.array([[1, 2], [2, 2], [2, 3],
             [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
labels = clustering.labels_
print(clustering.labels_)

unique, counts = np.unique(labels, return_counts=True)
bigger_cluster_label_idx = counts.argmax()
bigger_cluster_label = unique[bigger_cluster_label_idx]
flt = np.where(clustering.labels_ == bigger_cluster_label, True, False)
points[flt]

print(X)
print(X[flt])
