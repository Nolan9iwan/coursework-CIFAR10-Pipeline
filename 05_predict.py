from tensorflow.keras.datasets import cifar10
from model_03 import model 

# Load only the test data
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize the test images
x_test = x_test / 255.0

import numpy as np

#prediction
predictions = model.predict(x_test)

# Convert prediction vectors to label numbers (0–9)
predicted_labels = np.argmax(predictions, axis=1)