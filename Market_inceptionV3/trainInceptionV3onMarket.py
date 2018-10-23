from keras import Model, optimizers
from keras.layers import AveragePooling2D, Flatten, Dense

from Market_VGG.trainVGGonMarket import count_id, create_ytrain, create_xtrain, halfGPU
from keras.applications.inception_v3 import InceptionV3


def create_inceptionV3_model(n_classes, n_layers_to_freeze):
    # @input: num of classes of the new final layer and num of layers to freeze
    # @output: InceptionV3 final model with new softmax layer at the end
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

    # Fully Connected Softmax Layer
    x = base_model.output
    x_fc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x_fc = Flatten(name='flatten')(x_fc)
    x_fc = Dense(n_classes, activation='softmax', name='predictions')(x_fc)

    #creating final model
    final_model = Model(base_model, x_fc)
    print final_model.summary()

    # freeze the weights of first layers
    for layer in final_model.layers[:n_layers_to_freeze]:
        layer.trainable = False

    print final_model.summary()

    return final_model


def fine_tune_model(model, nb_epoch, batch_size, X_train, Y_train, lr):
    # @input: model to fine tune, num of epochs, batch size, train data and labels
    # @output: fine tuned model

    # compile the model with SGD and a very slow learning rate.
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=lr, momentum=0.9), metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    return model


halfGPU()

#path of market dataset
path='/media/data/dataset/Market-1501-v15.09.15/bounding_box_train/'
dictionary = count_id(path)

num_ID = len(dictionary)
print num_ID


X_train = create_xtrain(299, 299)
Y_train = create_ytrain(dictionary)

batch_size = Y_train.shape[0]/1000
print 'batch size: ' + str(batch_size)
nb_epoch = 2


model = create_inceptionV3_model(num_ID, 10) #10 DA CAMBIARE
fine_tune_model(model,nb_epoch,batch_size, X_train, Y_train)
model.save('/home/jansaldi/models/InceptionV3_Market.h5')

