import numpy as np
from PIL import Image
import itertools
import sys

# Number of clusters
k = 8

# Original image
print('Reading image...')
img = Image.open("./20col.png")
X = np.array(img).astype(np.float32)
h, w, c = X.shape
unique_colors, colors_count = np.unique(X.reshape(-1, 3), axis=0, return_counts=True)
N = unique_colors.shape[0]

# Loss threshold
if len(sys.argv) > 1:
    TARGET_OBJ = float(sys.argv[1])
else:
    try:
        lloyd = Image.open('./8col.png')
        lloyd = np.array(lloyd).reshape(-1, 3).astype(np.float64)
        TARGET_OBJ = np.sum((lloyd - np.array(img).reshape(-1, 3).astype(np.float64))**2)  # best objective value in k-means
        print(f'Loss threshold set to {TARGET_OBJ:,.2f} from lloyd-generated image.')
    except:
        TARGET_OBJ = 300e6
        print(f'Cannot find lloyd-generated image. Setting loss threshold to {TARGET_OBJ:,.2f}')


# -------------------------------------- #
# ----- Generate centroids ------------- #
# -------------------------------------- #
print('Generating centroids...')
centroids = []
centroids_obj = []
centroids_subset = []

subset_msk = itertools.product([0, 1], repeat=len(unique_colors))
for i, msk in enumerate(subset_msk):
    
    if i % 1e5 == 0:
        print(f'Iteration: {i:,} / {2**len(unique_colors):09,}')
    
    subset_cardinality = sum(msk)
    
    # Empty set
    if subset_cardinality == 0:
        continue
    
    # Suboptimal clustering
    if subset_cardinality >= (N-k+2):
        continue
    
    subset = unique_colors[np.array(msk) == 1]
    subset_counts = colors_count[np.array(msk) == 1]
    # centroid = subset.mean(axis=0)
    centroid = np.sum(subset * subset_counts.reshape(-1, 1), axis = 0) / subset_counts.sum()
    subset_obj = np.sum((subset - centroid)**2, axis=1).dot(subset_counts)

    # If already above target, skip
    if TARGET_OBJ < subset_obj:
        continue

    centroids.append(centroid)
    centroids_obj.append(subset_obj)
    centroids_subset.append(msk)
    
print(f'Centroids found: {len(centroids):,}')

centroids = np.array(centroids)
centroids_obj = np.array(centroids_obj)
centroids_subset = np.array(centroids_subset)

# -------------------------------------- #
# ----- Write to main.dat -------------- #
# -------------------------------------- #

print('Writing main.dat...')
path = 'main.dat'
k = 8

with open(path, 'w') as file:
    
    # Colors
    file.write('set I :=')
    for i in range(len(unique_colors)):
        file.write(f' {i+1}')
    file.write(';\n\n')

    # Centroids
    file.write('set J :=')
    for j in range(len(centroids)):
        file.write(f' {j+1}')
    file.write(';\n\n')

    # Max number of centroids
    file.write(f'param k := {k};\n\n')

    # Matrix X
    file.write('param X :')
    for j in range(len(centroids_subset)):
        file.write(f' {j+1}')
    file.write(' :=\n')
    for i in range(len(unique_colors)):
        file.write(f'{i+1}')
        for j in range(len(centroids_subset)):
            file.write(f' {centroids_subset[j,i]}')
        if i!= len(unique_colors)-1:
            file.write('\n')
    file.write(';\n\n')

    # Cost of centroid j
    file.write('param d :=\n')
    for j in range(len(centroids_obj)):
        file.write(f' {j+1} {centroids_obj[j]}')
        if j!= len(centroids)-1:
            file.write('\n')
    file.write(';\n\n')
print('Done!')

# Save to reconstruct image with IP-solution
np.save('./centroids.npy', centroids)
np.save('./centroids_subset.npy', centroids_subset)
np.save('./unique_colors.npy', unique_colors)