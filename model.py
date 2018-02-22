from load_data import load_data, read_csv, generator
from sklearn.model_selection import train_test_split

# Argument passing
import argparse

parser = argparse.ArgumentParser(description="Run your own configuration")
parser.add_argument(
    '-g',
    '--generator',
    type=bool,
    default=False,
    help="Use generator if True (less memory intensive)"
)
parser.add_argument(
    '-e',
    '--nb_epoch',
    type=int,
    nargs='?',
    default=2,
    help="Define the number of epoch: default 2"
)
parser.add_argument(
    '-c',
    '--continue_training',
    type=bool,
    nargs='?',
    default=False,
    help="Continue from previous train model"
)
args = parser.parse_args()

# load pre-processed data
if args.generator == True:
    # load pre-processed data with generator
    samples, data_info_hash = read_csv()
    train_samples, validation_samples = train_test_split(samples, test_size=0.3)
    training_generator = generator(train_samples)
    validation_generator = generator(validation_samples)
    # display data metrics
    input_shape = data_info_hash["input_shape"]
    print("> Number of samples:", data_info_hash["num_samples"])
    print("> Sample Image shape:", input_shape)
else:
    # load pre-processed data without generator
    X_train, y_train = load_data()

    print("> Number of samples:", X_train.shape[0])
    input_shape = X_train[0].shape
    print("> Sample Image shape:", input_shape)
    output_shape = y_train.shape
    print("> Output shape:", output_shape)

# Neural Network
model_path = 'model.h5'
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
# Neural Network - Regression Network
if args.continue_training == False:
    print("> Building Neural Network")
    model = Sequential()
    # Pre-process: normalize and mean-center
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    # Pre_process: Crop out the sky and car hood
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    print("> Trainning Starting:")
    # Use Mean Squared Error(mse) as loss function
    # Use Adam Optimizer as optimization function
    model.compile(loss='mse', optimizer='adam')
else:
    print("> Continue training Neural Network")
    from keras.models import load_model
    model = load_model(model_path)

# Split 20% of training data as validation data
nb_epoch = args.nb_epoch
if args.generator == True:
    model.fit_generator(nb_epoch=nb_epoch, generator=training_generator, samples_per_epoch=len(train_samples) * 6, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 6)
else:
    model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=nb_epoch)

model.save(model_path)
print("> Model saved to:", model_path)
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)
