import pandas as pd
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import os

def getTrust(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Trustworthiness[i]

rawData = pd.read_csv('ratings.csv')
print("Shape of the raw data", rawData.shape)

# images = []
# labels = []

mFile = open("train_labels.txt", "w")

for filename in os.listdir("./train_images/"):
	if filename.endswith(".png"):
		mFile.write(filename + "," + str(getTrust(filename)) + "\n")
		

# for i in range(rawData.shape[0]):
# 	img_name = rawData.Face[i]
# 	trustworthiness = rawData.Trustworthiness[i]
# 	images.append(imread("./images500/"+img_name+".png"))
# 	labels.append(trustworthiness)


# print("img name: " + rawData.Face[10])
# print(labels[10])
# plt.imshow(images[10])
# plt.show()