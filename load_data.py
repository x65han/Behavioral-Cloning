import csv, cv2
import numpy as np

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

def load_data():
    # Data Preparation
    lines = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for each_line in reader:
            lines.append(each_line)

    images = []
    correction = 0.1
    measurements = []
    print("Loading Images")
    # for line in lines:
    for w in range(1000):
        line = lines[w]
        # load left, right and center image
        for index in range(3):
            image_path = "./data/" + line[index].strip()
            image = cv2.imread(image_path)
            images.append(image)
            measurement = float(line[3])
            # add correction to left image
            if index == 1: measurement = measurement + correction
            # add correction to right image
            if index == 2: measurement = measurement - correction
            measurements.append(measurement)

    # Augment Data
    print("Augmenting Data")
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        # gray scale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # crop out sky and car hood
        image = image[75:-25, :]
        # save normal image
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        # flip image horizontally
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    # visualize_data(augmented_images, augmented_measurements, title="after", gray_scale=True)

    return np.array(augmented_images), np.array(augmented_measurements)