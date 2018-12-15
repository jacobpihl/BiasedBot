import pandas as pd
import numpy as np
import cv2
import os

IMAGE_PATH = "./train_images/"
BACKGROUND_PATH = "./backgrounds/"
SAVE_PATH = "./augmented_train_images/"


def getAttr(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Attractiveness[i]


def getMood(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Mood[i]


def getTrust(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Trustworthiness[i]


def getMasc(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Masculinity[i]


def getAge(filename):
	for i in range(rawData.shape[0]):
		img_name = rawData.Face[i]
		if img_name == os.path.splitext(filename)[0]:
			return rawData.Age[i]


def augment(facename, bgname):
	# Read images
	im = cv2.imread(IMAGE_PATH + facename)
	bg = cv2.imread(BACKGROUND_PATH + bgname)
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Got two different BG colors so 2 different processes
	ret, thresh = cv2.threshold(imgray, 250, 255, cv2.THRESH_BINARY)
	if np.sum(thresh) < 182488:
		ret, thresh = cv2.threshold(imgray, 202, 255, cv2.THRESH_BINARY)

	# Draw big contours, aka fill in the face part of the image
	# Use this as mask
	mask = np.zeros(thresh.shape, dtype=np.uint8)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		if cv2.contourArea(c) > 5000:
			cv2.drawContours(mask, [c], 0, (255,255,255), -1)

	# Mask the face (cut it out)
	inv_mask = cv2.bitwise_not(mask)
	masked_out_face = cv2.bitwise_and(im, im, mask=inv_mask)

	# # Mask background inversly of the face
	masked_bg = cv2.bitwise_and(bg, bg, mask=mask)

	# # Put together new image
	new_im = cv2.bitwise_or(masked_bg, masked_out_face)

	#cv2.imshow("res", new_im)
	#cv2.waitKey(5)

	return new_im



if __name__ == "__main__":
	c = 0	# For appending a number at the end of each augmented image

	rawData = pd.read_csv('ratings.csv')
	mFile = open("train_labels.txt", "w")

	print("Starting")
	for face in os.listdir("./train_images/"):
		c = 0

		# Add original image
		if face.endswith(".png"):
			im = cv2.imread(IMAGE_PATH + face)
			cv2.imwrite(SAVE_PATH + face, im)
			mFile.write(face + "," + str(getAttr(face)) + "," + str(getMood(face)) + "," + str(getTrust(face)) + "," + str(getMasc(face)) + "," + str(getAge(face)) + "\n")

		for bg in os.listdir("./backgrounds/"):
			if face.endswith(".png") and bg.endswith(".png"):
				# Augment background
				aug_img = augment(face, bg)

				# path creation stuff
				ending = face[-4::]
				img_name = face[:-4:] + '_' + str(c) + ending

				# Save image and rating
				cv2.imwrite(SAVE_PATH + img_name, aug_img)
				mFile.write(img_name + "," + str(getAttr(face)) + "," + str(getMood(face)) + "," + str(getTrust(face)) + "," + str(getMasc(face)) + "," + str(getAge(face)) + "\n")

				c += 1

				# u = cv2.waitKey(5)
				# while u != ord("q"):
				# 	u = cv2.waitKey(5)
	print("Done.")