import os,random
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import scipy
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

input_size=(224,224,3)
train_dir="./train&test/train"
test_dir="./train&test/test"
batch_size=32
epochs_max=20

def add_noise(img):
    varibility=50
    deviation=varibility*random.random()
    noise=np.random.normal(0,deviation,img.shape)
    img=img+noise
    np.clip(img,0.,255.)
    return img

def show_train(history,train,validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

generator = ImageDataGenerator()
classes=generator.flow_from_directory('./train&test/train/')
classes=classes.class_indices
print(len(classes))

img_generator=ImageDataGenerator(preprocessing_function=preprocess_input)
augmentation_generator=ImageDataGenerator(preprocessing_function=add_noise,
                                          rotation_range=30,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          shear_range=0.1,
                                          zoom_range=0.1,
                                          horizontal_flip=True,
                                          fill_mode='nearest',
                                        )
#training set
train_data=augmentation_generator.flow_from_directory(train_dir,
                                                      target_size=(224,224),
                                                      shuffle=True,
                                                      batch_size=16,
                                                      seed=100,
                                                      class_mode='categorical',
                                                      classes=classes
                                                    )

#validation set
valid_data=img_generator.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             shuffle=False,
                                             batch_size=16,
                                             seed=100,
                                             class_mode='categorical',
                                             classes=classes
                                            )

#model structure
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
#建立後段的分類層，兩個全連接，最後softmax
flatten = tf.keras.layers.Flatten(name='flatten')(model.output)
fc1 = tf.keras.layers.Dense(4096, activation='relu',name='fc1')(flatten)
fc2 = tf.keras.layers.Dense(4096, activation='relu',name='fc2')(fc1)
out = tf.keras.layers.Dense(300, activation='softmax')(fc2)
model = tf.keras.Model(model.input, out)
model.summary()
#optimizer嘗試使用adam及sgd做為測試
adm = tf.keras.optimizers.Adam(learning_rate=0.0001)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = adm, loss = 'categorical_crossentropy', metrics = ['accuracy'])


#有中斷訓練，因此將模型載回來，從斷點繼續Train.
#model.load_weights('best1.h5')


#兩個callback機制，存val_accuracy最高的模型以及沒有再收斂提前中斷的機制
cb_best=tf.keras.callbacks.ModelCheckpoint('best.h5',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1
                                          )
cb_stop=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=5,
                                         verbose=1)

history=model.fit(
    train_data,
    validation_data=valid_data,
    batch_size=None,
    epochs=epochs_max,
    callbacks=[cb_best,cb_stop],
    verbose=1
)



show_train(history,'accuracy','val_accuracy')
show_train(history,'loss','val_loss')