import os, imageio
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
from skimage.transform import resize
from PIL import Image

def readFile(path):
    fp = open(path)
    lines = fp.readlines()
    res = np.array([line.strip() for line in lines])
    return shuffle(res)
    

def normalize(res):
    res = np.float32(res)
    res = np.divide(res, 127.5)
    res = np.subtract(res, 1)
    return res

def denormalize(data):
    res = np.add(res, 1)
    res = np.multiply(res, 127.5)
    res = res.astype(np.uint8)
    return res

def preprocess_feature_img(img, target_size, normalize=True):
    pad_h = max(target_size[0] - img.shape[0], 0)
    pad_w = max(target_size[1] - img.shape[1], 0)
    # pad image with black pixels
    # img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)), 'constant', constant_values=0.)
    res = cv.resize(img, (target_size[0], target_size[1]))
    res = res.astype(np.uint8)

    # Normalize
    if normalize is True:
        res = np.float32(res)
        res = np.divide(res, 127.5)
        res = np.subtract(res, 1)
    
    return res


def preprocess_label_img(img, target_size, classes=21, one_hot=True):
    pad_h = max(target_size[0] - img.shape[0], 0)
    pad_w = max(target_size[1] - img.shape[1], 0)
    # pad image with black pixels
    # img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), 'constant', constant_values=0.)
    img = cv.resize(img, (target_size[0], target_size[1]))
    img = img.astype(np.uint8)

    if one_hot is False:
        return img

    # One hot encoding
    res = np.zeros((target_size[0], target_size[1], classes))
    for c in range(classes):
        res[:, :, c] = (img == c).astype(np.uint8)

    return res


def get_file_size(path):
    data_name_list = readFile(path)
    num_data = data_name_list.shape[0]
    return num_data


def seg_to_palette(img):
  img = np.argmax(img, axis=-1).astype(np.uint8)
  res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  res[img[:,:] ==  0] = [  0,   0,   0]
  res[img[:,:] ==  1] = [128,   0,   0]
  res[img[:,:] ==  2] = [  0, 128,   0]
  res[img[:,:] ==  3] = [128, 128,   0]
  res[img[:,:] ==  4] = [  0,   0, 128]
  res[img[:,:] ==  5] = [128,   0, 128]
  res[img[:,:] ==  6] = [  0, 128, 128]
  res[img[:,:] ==  7] = [128, 128, 128]
  res[img[:,:] ==  8] = [ 64,   0,   0]
  res[img[:,:] ==  9] = [192,   0,   0]
  res[img[:,:] == 10] = [ 64, 128,   0]
  res[img[:,:] == 11] = [192, 128,   0]
  res[img[:,:] == 12] = [ 64,   0, 128]
  res[img[:,:] == 13] = [192,   0, 128]
  res[img[:,:] == 14] = [ 64, 128, 128]
  res[img[:,:] == 15] = [192, 128, 128]
  res[img[:,:] == 16] = [  0,  64,   0]
  res[img[:,:] == 17] = [128,  64,   0]
  res[img[:,:] == 18] = [  0, 192,   0]
  res[img[:,:] == 19] = [128, 192,   0]
  res[img[:,:] == 20] = [  0,  64, 128]
  return res
    
def get_unique_rgb(img):
    return set( tuple(v) for m2d in img for v in m2d )


def generator(
    data_list_path="~/.keras/datasets/VOC2012/combined_imageset_train.txt",
    # data_list_path="~/.keras/datasets/VOC2012/combined_imageset_val.txt",
    feature_data_path='~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages',
    label_data_path='~/.keras/datasets/VOC2012/combined_annotations',
    batch_size=16,
    classes=21,
    feature_extension='.jpg',
    label_extension='.png',
    target_size=(320, 320),
    normalize=True,
):
    if '~' in data_list_path: data_list_path = os.path.expanduser(data_list_path)
    if '~' in feature_data_path: feature_data_path = os.path.expanduser(feature_data_path)
    if '~' in label_data_path: label_data_path = os.path.expanduser(label_data_path)

    data_name_list = readFile(data_list_path)
    num_data = data_name_list.shape[0]
    print(f'[Generator] Fetched a list of {num_data} data.')
    print(f'[Generator] normalize = {normalize}')

    while True:
        for offset in range(0, num_data, batch_size):
            batch_name_list = data_name_list[offset:offset + batch_size]

            batch_x = np.zeros((batch_size, ) + target_size + (3,), dtype=np.float32)
            batch_y = np.zeros((batch_size, target_size[0], target_size[1], classes), dtype=np.uint8)

            for i in range(batch_size):
                if offset + i >= num_data: break

                # Fetch single train data
                x = imageio.imread(os.path.join(feature_data_path, batch_name_list[i] + feature_extension))
                batch_x[i] = preprocess_feature_img(x, target_size, normalize=normalize)

                # Fetch single label data
                PIL_img = Image.open(os.path.join(label_data_path, batch_name_list[i] + label_extension))
                y = np.array(PIL_img)
                y[np.where(y == 255)] = 21
                batch_y[i] = preprocess_label_img(y, target_size, classes)

            yield batch_x, batch_y


if __name__ == '__main__':
    gen = generator()
    next(gen)
