import csv, cv2
import numpy as np
from sklearn.utils import shuffle

# Visualize data
import matplotlib.pyplot as plt
import random

def visualize_data(X_data, y_data, title="No Title", gray_scale=False, limit=15, isRand=True):
    fig, axs = plt.subplots(3, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in range(15):
        axs[i].axis('off')

    for i in range(limit):
        index = i
        if isRand == True:
            index = random.randint(0, len(X_data) - 1)
        image = X_data[index]
        if gray_scale == True:
            axs[i].imshow(image.squeeze(), cmap='gray')
        else:
            axs[i].imshow(image)
        axs[i].set_title(y_data[index])
    fig.canvas.set_window_title(title)
    plt.show()

def pre_process(X_data, multiple=True):
    # crop out the sky and car hood
    print("> Cropping Image")
    if multiple==True:
        X_data = X_data[:, 75:-25]
    else:
        X_data = X_data[75:-25]
    # normalize images
    print("> Normalizing Image")
    X_data = X_data / 255.0 - 0.5
    # gray scale images
    print("> Gray Scaling Image")
    if multiple==True:
        X_data = np.sum(X_data / 3, axis=3, keepdims=True)
    else:
        X_data = np.sum(X_data / 3, axis=2, keepdims=True)

    return X_data

def augment_data(X_data, y_data):
    X_output, y_output = [], []
    for image, measurement in zip(X_data, y_data):
        # save normal image
        X_output.append(image)
        y_output.append(measurement)
        # flip image horizontally if magnitude is > 0.33
        if abs(measurement) > 0.2:
            flipped_image = cv2.flip(image, 1)
            X_output.append(flipped_image)
            y_output.append(measurement * -1)

    return np.array(X_output), np.array(y_output)

def read_csv():
    # Data Preparation
    lines = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the headers
        for each_line in reader:
            lines.append(each_line)

    # gather data metrics
    hash = {}
    hash["num_samples"] = len(lines)
    hash["input_shape"] = cv2.imread("./data/" + lines[0][0].strip()).shape
    return lines, hash

def generator(samples, batch_size=1):
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # Load Images
            images = []
            correction = 0.1
            measurements = []
            # print("> Loading Images")
            for line in batch_samples:
                # load left, right and center image
                for index in range(3):
                    image_path = "./data/" + line[index].strip()
                    image = cv2.imread(image_path)
                    images.append(image)
                    measurement = float(line[3])
                    # if index == 1: # add correction to left image
                    #     measurement = measurement + correction
                    # elif index == 2: # add correction to right image
                    #     measurement = measurement - correction
                    # else:
                    measurement *= 6
                    measurements.append(measurement)

            # convert to numpy array
            images = np.array(images)
            measurements = np.array(measurements)
            # pre process images
            # images = pre_process(images)
            # Augment Data
            images, measurements = augment_data(images, measurements)
            # Visualize Data
            # visualize_data(images, measurements, title="Images", gray_scale=True)

            yield shuffle(images, measurements)


def load_data():
    # Data Preparation
    lines, _ = read_csv()

    # Load Images
    images = []
    correction = 0.2
    measurements = []
    print("> Loading Images")
    for line in lines:
        # load left, right and center image
        for index in range(3):
            image_path = "./data/" + line[index].strip()
            image = cv2.imread(image_path)
            images.append(image)
            measurement = float(line[3])
            if index == 1: # add correction to left image
                measurement = measurement + correction
            elif index == 2: # add correction to right image
                measurement = measurement - correction
            else:
                measurement *= 1
            measurements.append(measurement)

    # convert to numpy array
    images = np.array(images)
    measurements = np.array(measurements)
    # pre process images
    # images = pre_process(images)
    # Augment Data
    print("> Augmenting Data")
    images, measurements = augment_data(images, measurements)
    # Visualize Data
    # visualize_data(images, measurements, title="Images", gray_scale=True)

    return images, measurements
