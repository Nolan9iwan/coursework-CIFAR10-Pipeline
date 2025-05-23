from tensorflow.keras.datasets import cifar10
from model_03 import model  
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the image data 
x_train = x_train / 255.0
x_test = x_test / 255.0

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

