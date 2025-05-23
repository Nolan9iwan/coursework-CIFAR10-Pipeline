from tensorflow.keras.datasets import cifar10 
# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print the shapes to confirm it's working
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)
