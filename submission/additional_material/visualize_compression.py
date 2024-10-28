import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


centroids = np.load('./centroids.npy').astype(np.uint8)
centroids_subset = np.load('./centroids_subset.npy').astype(bool)
unique_colors = np.load('./unique_colors.npy').astype(np.uint8)

with open('out.txt', 'r') as f:
    f = f.read()
    chosen_centroids = re.split(r'obj.val.+', f)[1]
    chosen_centroids = re.split(r'Model has been.+', chosen_centroids)[0]
    chosen_centroids = re.findall(r'\d+', chosen_centroids)
    chosen_centroids = [int(i)-1 for i in chosen_centroids]

# Load image and keep np.uint8 format for comparison
img = Image.open('20col.png')
X = np.array(img)
h, w, c = X.shape
X = X.reshape(-1, c)

for i in range(len(chosen_centroids)):
    
    # which colors are in the cluster
    cluster_msk = centroids_subset[chosen_centroids[i]]
    original_color = unique_colors[cluster_msk]
    
    # new color for the cluster
    new_color = centroids[chosen_centroids][i]
    for j in range(len(original_color)):
        X[np.all(X == original_color[j], axis=1)] = new_color

img_compressed = Image.fromarray(X.reshape(h, w, c).astype(np.uint8), mode='RGB')
img_compressed.save('./8col_ip.png')

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 3, 1)
plt.axis('off')
img = np.array(img).reshape(-1, 3).astype(np.float32)
img_size = os.path.getsize('./20col.png')
plt.title(f'Original Image\nloss = {0:,.2f}\nsize = {img_size/1e3:,.2f} KB\n')
plt.imshow(Image.open('./20col.png'))

# Lloyd-compressed image
plt.subplot(1, 3, 2)
plt.axis('off')
lloyd = np.array(Image.open('./8col.png')).reshape(-1, 3).astype(np.float32)
lloyd_loss = np.sum((lloyd - img)**2)
lloyd_size = os.path.getsize('./8col.png')
plt.title(f'Lloyd-compressed Image\nloss = {lloyd_loss:,.2f}\nsize = {lloyd_size/1e3:,.2f} KB\n')
plt.imshow(Image.open('./8col.png'))

# IP-compressed image
plt.subplot(1, 3, 3)
plt.axis('off')
ip = X.reshape(-1, 3).astype(np.float32)
ip_loss = np.sum((ip - img)**2)
ip_size = os.path.getsize('./8col_ip.png')
plt.title(f'IP-compressed Image\nloss = {ip_loss:,.2f}\nsize = {ip_size/1e3:,.2f} KB\n')
plt.imshow(Image.open('./8col_ip.png'))

plt.savefig('./all_compressions.png')

plt.show()