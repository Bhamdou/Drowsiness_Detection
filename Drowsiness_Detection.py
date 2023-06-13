import os
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Dense, Flatten

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Limit TensorFlow to only use CPUs
cpus = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(cpus, 'CPU')

# Load the VGGFace model
base_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model with a new output layer
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
