import keras
from PIL import Image
from keras import optimizers, Input, Model
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, \
    Dense
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet50 import conv_block, identity_block

NUM_OF_CHARACTERS_OF_ID = 4
NORMALIZING_COSTANTS = [103.939, 116.779, 123.68]
GPU_FRACTION = 0.5
NUM_LAYERS_TO_FREEZE = 174
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
SHAPE_INPUT_NN = (224, 224, 3)
BATCH_SIZE = 16


def halfGPU():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    set_session(tf.Session(config=config))


def count_id(path):
    # @input : path of Market
    # @output : dictionary of ID-integers to build keras input labels
    listing = os.listdir(path)
    #dictionary = dictionary for conversion from ID to continuous mapping output (key :ID, integer from 0 to 750)
    id_int_dictionary = {}
    index = 0
    for filename in listing:
        if filename.endswith('.jpg'):
            # ID indice ID immagine
            ID = filename[:NUM_OF_CHARACTERS_OF_ID]
            if index == 0:
                id_int_dictionary[ID] = index
                index += 1
            else:
                is_not_already_listed = True
                for keys in id_int_dictionary:
                    if keys == ID:
                        is_not_already_listed = False
                if is_not_already_listed:
                    id_int_dictionary[ID] = index
                    index += 1
    return id_int_dictionary


def count_images(path_traindata):
    # @input : path of train images
    # @output : num of images in the directory
    listing = os.listdir(path_traindata)
    num_imgs = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            num_imgs += 1
    return num_imgs


def image_read(img_path):
    image = Image.open(img_path).convert('RGB')
    return image


def prepare_x_train(image):
    # preprocess img sistemare paremetri traindata
    x = img_to_array(image)
    x = x[:, :, ::-1]
    x[:, :, 0] -= NORMALIZING_COSTANTS[0]
    x[:, :, 1] -= NORMALIZING_COSTANTS[1]
    x[:, :, 2] -= NORMALIZING_COSTANTS[2]
    return x


def prepare_y_train(id_int_dictionary, filename, num_id):
    ID = filename[:NUM_OF_CHARACTERS_OF_ID]
    y = id_int_dictionary[ID]
    y = keras.utils.to_categorical(y, num_id)
    return y


def create_trainData(shape_input_nn, path_train, id_int_dict, num_id):
    # @input : size of images, path of training images
    # @output : X_train and Y_train
    listing = os.listdir(path_train)
    num_imgs = count_images(path_train)
    X_train = np.empty((num_imgs, shape_input_nn[0], shape_input_nn[1], shape_input_nn[2]), 'float32')
    Y_train = np.empty((num_imgs, num_id), 'float32')
    index = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            # create x_train
            image = image_read(path_train + filename).resize(shape_input_nn[0:2])
            X_train[index, :, :, :] = prepare_x_train(image)

            # create Y_train
            Y_train[index, :] = prepare_y_train(id_int_dict, filename, num_id)
            index += 1

    print "dimensione x_train: " + str(X_train.shape)
    print "dimensione y_train: " + str(Y_train.shape)
    return X_train, Y_train


def freeze_layers(model, n_layers_to_freeze):
    # freeze the weights of firsts layers
    for layer in model.layers[:n_layers_to_freeze]:
        layer.trainable = False
    return model


def create_Resnet_Model(num_classes, n_layers_to_freeze):
    # @input: num of classes of the new final softmax layer, num layers to freeze
    # @output: Resnet final model with new softmax layer at the end

    #creating Resnet network
    img_input = Input(shape=SHAPE_INPUT_NN)
    bn_axis = 3
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fcnew')(x_fc)

    resnet_model = Model(img_input, x_fc)

    # load weights
    resnet_model.load_weights('/home/jansaldi/Progetto-tesi/Market_Resnet/weights/resnet50_tf_weights_imagenet.h5')

    #creating new last softmax layer
    x_new_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_new_fc = Flatten()(x_new_fc)
    x_new_fc = Dense(num_classes, activation='softmax', name='fcnew')(x_new_fc)

    #creating the new model
    resnet_model = Model(img_input, x_new_fc)

    #freeze layers
    resnet_model = freeze_layers(resnet_model, n_layers_to_freeze)

    print resnet_model.summary()
    return resnet_model


def fine_tune_model(model_to_fine_tune, nb_epoch, batch_size, traindata, learning_rate):
    # compile the model with SGD and a very slow learning rate
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
    model_to_fine_tune.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model_to_fine_tune.fit(traindata[0], traindata[1], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    return model_to_fine_tune


halfGPU()

# path of market
path_traindata='/media/data/dataset/Market-1501-v15.09.15/bounding_box_train/'
dictionary = count_id(path_traindata)

num_ID = len(dictionary)
print "num of identities: " + str(num_ID)

traindata = create_trainData(SHAPE_INPUT_NN, path_traindata, dictionary, num_ID)

model = create_Resnet_Model(num_ID, NUM_LAYERS_TO_FREEZE)
model = fine_tune_model(model, NUM_EPOCHS, BATCH_SIZE, traindata, LEARNING_RATE)
model.save('/home/jansaldi/models/Resnet_Market.h5')
