from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
from scipy import misc

from flask import Flask
from flask import request, render_template

app = Flask(__name__)
model = None
modelLoaded = False

@app.route('/', methods=['GET'])
def index():
	return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
	global modelLoaded
	global model
	if request.method == 'POST':
		print("IN POST")
		photo = request.files['photo']
		photo.save("temp.png")
		photo.close()

		img = misc.imread("temp.png", mode="RGB")
		resized_image = misc.imresize(img, (500,500))
		resized_image = resized_image / 255.0
		if not modelLoaded:
			model = load_model()
			modelLoaded = True

		nin = np.expand_dims(resized_image, 0)
		pred = model.predict(nin)
		print("Predicted: ")
		print(pred)
		return str(pred[0][0])
		
	return render_template("index.html")


def load_model():
	# Load model
	json_file = open('./model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = keras.models.model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("./model.h5")
	print("Loaded model from disk")

	# Using same settings as other file to compile (but keras optimizer)
	optimizer = keras.optimizers.Adam()
	loaded_model.compile(loss='mse',
					  optimizer=optimizer,
					  metrics=['mae'])
	print("Compiled model")
	return loaded_model

if __name__ == '__main__':
	app.run()
