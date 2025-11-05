"""
AlexNet implementation
Adapted for smaller images (MNIST, Fashion-MNIST, CIFAR)
"""
import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel, ConvBlock, DenseBlock
from activation_functions.activations import get_activation


class AlexNet(BaseModel):
    """
    AlexNet architecture (adapted for small images)
    Original paper: ImageNet Classification with Deep Convolutional Neural Networks
    """
    
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation)
        
    def build_model(self):
        """Build AlexNet model"""
        
        # Get activation function
        act_fn = get_activation(self.activation)
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Resize if input is too small (for MNIST/Fashion-MNIST)
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        # Convert grayscale to RGB if needed
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        # Conv Block 1
        x = keras.layers.Conv2D(96, 11, strides=4, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(3, strides=2)(x)
        
        # Conv Block 2
        x = keras.layers.Conv2D(256, 5, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(3, strides=2)(x)
        
        # Conv Block 3
        x = keras.layers.Conv2D(384, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # Conv Block 4
        x = keras.layers.Conv2D(384, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # Conv Block 5
        x = keras.layers.Conv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(3, strides=2)(x)
        
        # Flatten
        x = keras.layers.Flatten()(x)
        
        # FC layers
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='AlexNet')
        return self.model


class AlexNetSmall(BaseModel):
    """
    Smaller version of AlexNet for faster training
    """
    
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation)
        
    def build_model(self):
        """Build smaller AlexNet model"""
        
        act_fn = get_activation(self.activation)
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Resize if needed
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        # Convert to RGB if grayscale
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        # Reduced filter sizes
        x = keras.layers.Conv2D(64, 5, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(2, strides=2)(x)
        
        x = keras.layers.Conv2D(128, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(2, strides=2)(x)
        
        x = keras.layers.Conv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.Conv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(2, strides=2)(x)
        
        x = keras.layers.Flatten()(x)
        
        # Reduced FC layer sizes
        x = keras.layers.Dense(1024)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='AlexNet-Small')
        return self.model