import numpy as np
import matplotlib.pyplot as plt

from data_treat import load_data_for_plot

malhas = ['mesh2d-1', 'mesh00000001-2d', 'mesh00000002-2d', 'bfs-mesh-1-2d', 'bfs-mesh-2-2d', 'mesh00000003-2d'] 
path = "../data_stencils/"

features, weights = load_data_for_plot(malhas, path)

features = np.array(features) #[x, y, d, w]
weights = np.array(weights)
print(np.shape(features))
print(np.shape(weights))

plt.subplot(221)
plt.scatter(features[:,3], weights)
plt.xlabel("1/dist")
plt.ylabel("peso")
plt.subplot(222)
plt.scatter(features[:,0], weights)
plt.xlabel("x")
plt.ylabel("peso")
plt.subplot(223)
plt.scatter(features[:,1], weights)
plt.xlabel("y")
plt.ylabel("peso")
plt.subplot(224)
plt.scatter(features[:,2], weights)
plt.xlabel("dist")
plt.ylabel("peso")
plt.show()