import six
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os, shutil, getpass

class imageAnalysis:

  imageOutputDir = 'output/'

  def __init__(self):
    print('constructor.....')
    self.cleanDir(self.imageOutputDir)
    self.makeDirWriteable(self.directory)

  def run(self, directory, number_of_colors = 3):
    self.number_of_colors = number_of_colors
    self.directory = self.preparePath(directory)
    
    for filename in os.listdir(directory):
      if filename.endswith(".png"):
          self.filename = os.path.splitext(filename)[0]
          self.fullPath = directory + filename
          self.read(directory + filename)
          self.resize()
          self.detectColours()
          self.printPie(self.imageOutputDir, self.filename)
      continue

  def RGB2HEX(self, color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

  def preparePath(self, directory):
    return os.path.join(directory, '')

  def makeDirWriteable(self, directory):
    os.chmod(directory, 0o777)
  
  def cleanDir(self, directory):
    shutil.rmtree(directory, True)
    if not os.path.exists(self.imageOutputDir):
      os.makedirs(self.imageOutputDir,0o777)

  def read(self, imagepath):
    print(imagepath)
    imageRead = cv2.imread(imagepath)
    self.image = cv2.cvtColor(imageRead, cv2.COLOR_BGR2RGB)

  def imageInfo(self):
    print("The type of this input is {}".format(type(self.image)))
    print("Shape: {}".format(self.image.shape))

  def resize(self):
    modified_image = cv2.resize(self.image, (600, 400), interpolation = cv2.INTER_AREA)
    self.modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

  def detectColours(self):
    clf = KMeans(n_clusters = self.number_of_colors)
    labels = clf.fit_predict(self.modified_image)
    self.counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in self.counts.keys()]
    self.hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in self.counts.keys()]
    self.rgb_colors = [ordered_colors[i] for i in self.counts.keys()]

  def printPie(self, outputDir, filename):
    plt.figure(figsize = (8, 6))
    plt.pie(self.counts.values(), labels = self.hex_colors, colors = self.hex_colors)
    graphFilename = outputDir + filename + '_pie.png'
    print(graphFilename)
    plt.savefig(graphFilename)