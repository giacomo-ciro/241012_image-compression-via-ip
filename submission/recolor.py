import sys
import numpy as np
from PIL import Image

# -------------------------------------- #
# ----- K-Means clustering class ------- #
# -------------------------------------- #

class KMeans:
    def __init__(self,
                 n_clusters=None,
                 init='random',
                 n_init=10,
                 tol = 1e-4,
                 normalize=False
                 )->None:

        if n_clusters is None:
            raise ValueError('Number of clusters must be specified.')
        
        self.n_clusters = n_clusters
        self.n_init = 1 if init == 'frequency' else n_init
        self.init = init
        self.tol = tol
        self.normalize = normalize

    def fit(self,
            X,
            max_iters=300
            )->None:
        """
        Fit the model to the data X.
        X is an array of shape (n_samples, n_features).
        Squared L-2 norm is used as the distance metric.
        All computations are done in float32.

        Args:
            X: np.array
                Data to fit the model on.
            max_iters: int
                Maximum number of iterations for one run of Lloyd's algorithm.

        """
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.normalize:
            X = X / 255
        
        best_obj = np.inf

        for _ in range(self.n_init):
            
            centroids, new_obj = self.lloyd(X, max_iters=max_iters)
            
            print(f'Iteration: {_+1:03} | Objective: {new_obj:,.2f}')
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_centroids = centroids
                print(f'New best objective: {best_obj:,.2f}')
        
        if self.normalize:
            best_centroids = best_centroids * 255
            X = X * 255

        self.centroids = best_centroids
        self.obj = best_obj
        self.labels = self.predict(X)
        
        print(f'Fitting done. Centroids stored in self. Objective value: {self.obj:,.2f}.')
        
        return
            
    def lloyd(self,
              X,
              max_iters
              ):
        """"
        One complete run of Lloyd's algorithm:
            1. Init centroids
            2. Assign each sample to the closest centroid
            3. Update the centroids 
        Returns the centroids and the objective value. All computations are done in float32.
        
        """
        if self.init == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        elif self.init in ['frequency', 'frequency+']:
            points, counts = np.unique(X, axis=0, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            centroids = points[sorted_indices[:self.n_clusters]]

            if self.init == 'frequency+':
                centroids += np.random.normal(0, 15, centroids.shape)
                centroids = centroids.clip(0, 255)
        else:
            raise ValueError('Invalid init method. Must be one of "random", "frequency" or "frequency+".')
        
        iter = 0
        while iter < max_iters:

            distances = np.sum((X[:, None] - centroids)**2, axis = 2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(self.n_clusters)])
            
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                centroids = new_centroids
                print(f'Converged in {iter} iterations.')
                break
            
            centroids = new_centroids
            iter += 1
        
        return centroids, np.min(distances, axis=1).sum()
        
    def predict(self,
                X
                ):
        """
        Return the closest centroid for each sample in X.
        X is an array of shape (n_samples, n_features).
        All computations are done in float32.

        """

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if getattr(self, 'centroids', None) is None:
            raise ValueError('Model not fitted.')
       
        return np.argmin(np.sum((X[:, None] - self.centroids)**2, axis = 2), axis=1)


# -------------------------------------- #
# ----- Get params from cmd ------------ #
# -------------------------------------- #

if len(sys.argv) != 4:
    print('Provide required arguments:\n\tpython recolor.py <input_img> <output_img> <k>')
    sys.exit(1)

input_img = sys.argv[1]     # path to input file
output_img = sys.argv[2]    # path to output file
k = int(sys.argv[3])        # number of colors


# -------------------------------------- #
# ----- Load and prepare image --------- #
# -------------------------------------- #

# Input image
img = Image.open(input_img).convert('RGB')

# Array of rgb pixels
img = np.array(img).astype(np.float32)
h, w, c = img.shape
X = img.reshape(h*w, c)

# Number of unique colors in the image
n_colors = np.unique(X, axis=0).shape[0]
print(f'Unique colors found in img: {n_colors}')
if k > n_colors:
    print(f'k={k} is greater than the number of unique colors in the image. Setting k to {n_colors}')
    k = n_colors

# -------------------------------------- #
# ----- Apply K-Means clustering ------- #
# -------------------------------------- #

print(f'Clustering data...')
kmeans = KMeans(n_clusters=k,
                init='frequency+',
                n_init=10,
                normalize=False
                )   # Adjust 'n_clusters' as needed
kmeans.fit(X)

# -------------------------------------- #
# ----- Convert back to RGB and save --- #
# -------------------------------------- #

X_compressed = kmeans.centroids[kmeans.labels]

print(f'Saving compressed image to {output_img}')
img_compressed = Image.fromarray(X_compressed.reshape(h, w, c).astype(np.uint8),
                                 mode='RGB'
                                 )
img_compressed.save(output_img)