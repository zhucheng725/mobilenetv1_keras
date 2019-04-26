
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

DESIRED_ACCURACY = 0.9


class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')> DESIRED_ACCURACY):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


model = keras.models.Sequential([

#Conv1
    keras.layers.Conv2D(32, (3,3), activation='relu',strides=(2, 2),padding='same',use_bias=False, input_shape=(224, 224, 3)),

#Conv2-3
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv4-5
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv6-7
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv8-9
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(256, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv10-11
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(256, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv12-13
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv13-22
    	keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    	keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    	keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
        keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
        keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(512, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv23-24
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(2, 2), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(1024, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv25-26
    keras.layers.DepthwiseConv2D(kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(1024, (1,1), strides=(1, 1), padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

#Conv27-28
    keras.layers.pooling.GlobalAveragePooling2D(),
    keras.layers.Dense(2, activation='softmax')
])


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc']
              )

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/media/zhu/1T/procedure/mobilenet/train_cat_dog',  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')
        #class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/media/zhu/1T/procedure/mobilenet/validation_cat_dog',  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')
        #class_mode='binary')


# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=2,  
#       epochs=50,
#       verbose=1,
#       callbacks=[callbacks])

history = model.fit_generator(
      train_generator,
      samples_per_epoch = 8000,
      #steps_per_epoch=80,
      validation_data = validation_generator,  
      epochs=500,
      verbose=1,
      callbacks=[callbacks])
    
