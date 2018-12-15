import pandas as pd
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import os

rawData = pd.read_csv('ratings.csv')

print("\nInformation about data")
print("Data shape: {}".format(rawData.shape))
print("Mean of trustworthiness: {}".format(rawData.Trustworthiness.mean()))
print("Standard Deviation of trustworthiness: {}".format(rawData.Trustworthiness.std()))
print("Minimum of trustworthiness: {}".format(rawData.Trustworthiness.min()))
print("Maximum of trustworthiness: {}".format(rawData.Trustworthiness.max()))

print("")
