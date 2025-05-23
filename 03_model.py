from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# define image shape and number of classes
input_shape = (32, 32, 3)  
n_classes = 10  

model = Sequential()

# 1st block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))

# 2nd block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# flatten and fully connected layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))  # can also try 128 later
model.add(Dropout(0.5))  # prevent overfitting
model.add(Dense(n_classes, activation='softmax'))  # final output layer

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print model summary to check structure
model.summary()






