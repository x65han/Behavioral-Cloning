from load_data import load_data

# load pre-processed data
X_train, y_train = load_data()

print("Number of samples:", X_train.shape[0])
input_shape = X_train[0].shape
print("Sample Image shape:", input_shape)
output_shape = y_train.shape
print("Output shape:", output_shape)

# Neural Network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
# Neural Network - Regression Network
print("Building Neural Network")
model = Sequential()
# Pre-process: normalize and mean-center 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
# Pre_process: Crop out the sky and car hood
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

print("Trainning Starting:")
# Use Mean Squared Error(mse) as loss function
# Use Adam Optimizer as optimization function
model.compile(loss='mse', optimizer='adam')
# Split 20% of training data as validation data
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=2)

model_path = 'model.h5'
model.save(model_path)
print("Model saved to:", model_path)
