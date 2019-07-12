from keras.layers import Activation, Add, Dropout, Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.models import Model

def FCN8(classes=None, target_size=None, VGG_Weights_path=None):
    input_height = target_size[0]
    input_width = target_size[1]
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    assert classes is not None
    assert target_size is not None
    assert VGG_Weights_path is not None

    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)  # (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)  # (None, 7, 7, 512)

    vgg = Model(img_input, pool5)
    # loading VGG weights for the encoder parts of FCN8
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same',
                name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same',
                    name="conv7", data_format=IMAGE_ORDERING))(o)

    # 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(classes, kernel_size=(4, 4),  strides=(
        4, 4), use_bias=False, data_format=IMAGE_ORDERING)(conv7)
    ## (None, 224, 224, 10)
    # 2 times upsampling for pool411
    pool411 = (Conv2D(classes, (1, 1), activation='relu', padding='same',
                      name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose(classes, kernel_size=(2, 2),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING))(pool411)

    pool311 = (Conv2D(classes, (1, 1), activation='relu', padding='same',
                      name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(classes, kernel_size=(8, 8),  strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)
    return model


def FCN16(classes=None, target_size=None, VGG_Weights_path=None):
    input_height = target_size[0]
    input_width = target_size[1]
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    assert classes is not None
    assert target_size is not None
    assert VGG_Weights_path is not None

    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)  # (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)  # (None, 7, 7, 512)

    vgg = Model(img_input, pool5)
    # loading VGG weights for the encoder parts of FCN8
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same',
                name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same',
                    name="conv7", data_format=IMAGE_ORDERING))(o)

    conv7_2 = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(conv7)

    pool411 = (Conv2D(classes, (1, 1), activation='relu', padding='same',
                      name="pool4_11", data_format=IMAGE_ORDERING))(pool4)

    o = Add(name="add")([pool411, conv7_2])
    o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(
        16, 16), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)
    return model


def FCN32(classes=None, target_size=None, VGG_Weights_path=None):
    input_height = target_size[0]
    input_width = target_size[1]
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    assert classes is not None
    assert target_size is not None
    assert VGG_Weights_path is not None

    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)  # (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)  # (None, 7, 7, 512)

    vgg = Model(img_input, pool5)
    # loading VGG weights for the encoder parts of FCN8
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same',
                name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same',
                    name="conv7", data_format=IMAGE_ORDERING))(o)

    # classifying layer
    conv7_score = (Conv2D(classes, (1, 1), activation='relu', padding='same',
                          name="pool3_11", data_format=IMAGE_ORDERING))(conv7)
    conv7_upscore = Conv2DTranspose(classes, kernel_size=(32, 32), strides=(
        32, 32), use_bias=False, data_format=IMAGE_ORDERING)(conv7_score)

    o = (Activation('softmax'))(conv7_upscore)
    model = Model(img_input, o)

    return model
