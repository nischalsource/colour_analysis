import six
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

#%matplotlib inline
show_chart = True
number_of_colors = 3

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

image = cv2.imread('sample.png')
imageOrig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#plt.imsave('newSample.png', imageOrig)
#plt.imsave('newSample3.png', gray_image, cmap='gray')
#plt.imsave('newSample2.png', image)

print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))

# RESIZE
modified_image = cv2.resize(imageOrig, (600, 400), interpolation = cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

# DETECT COLOURS
clf = KMeans(n_clusters = number_of_colors)
labels = clf.fit_predict(modified_image)

counts = Counter(labels)

center_colors = clf.cluster_centers_
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

# Print and Save a pie chart
if (show_chart):
    plt.figure(figsize = (8, 6))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    plt.savefig('pie.png')

print(rgb_colors)
