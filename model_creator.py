import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras


def build_regularized_cnn(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_regularized_cnn_l2_0001(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_ResNet50(input_shape=(224, 224, 3), num_classes=3, with_base_model=False):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    model = models.Sequential()
    model.add(base_model)
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    if with_base_model:
        return model, base_model
    return model

def build_cnn_medium(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_cnn_medium_less_normalization(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_cnn_medium_large_padding_same(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_cnn_mediapipe(input_shape=(64, 64, 1), num_classes=3):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
    
    
# U-Net model

def double_conv_layer(x, filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', batch_norm=False):
    x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x

def downsample_block(x, filters, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', batch_norm=False, dropout=0.0):
    x = double_conv_layer(x, filters, kernel_size, activation, padding, kernel_initializer, batch_norm)
    d = layers.MaxPooling2D(strides)(x)
    if dropout > 0.0:
        d = layers.Dropout(dropout)(d)
    return x, d

def upsampling_block(x, skip, filters, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', batch_norm=False, dropout=0.0):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.concatenate([x, skip])
    x = double_conv_layer(x, filters, kernel_size, activation, padding, kernel_initializer, batch_norm)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    return x

def build_unet_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)
    
    s1, d1 = downsample_block(inputs, 16, dropout=0.1)
    s2, d2 = downsample_block(d1, 32, batch_norm=True)
    s3, d3 = downsample_block(d2, 64, dropout=0.1)
    s4, d4 = downsample_block(d3, 128, batch_norm=True)
    
    bottleneck = double_conv_layer(d4, 256)
    
    u4 = upsampling_block(bottleneck, s4, 128, batch_norm=True)
    u3 = upsampling_block(u4, s3, 64, dropout=0.1)
    u2 = upsampling_block(u3, s2, 32, batch_norm=True)
    u1 = upsampling_block(u2, s1, 16, dropout=0.1)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u1)
    
    return models.Model(inputs=[inputs], outputs=[outputs])
